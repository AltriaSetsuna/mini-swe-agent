#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import random
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[6] / "outputs" / "agent"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
    "pro": "ScaleAI/SWE-bench_Pro",
    "deepswe": "datacurve/deep-swe",
    "rebenchv2":"nebius/SWE-rebench-V2"
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    dockerhub_tag = instance.get("dockerhub_tag", None)
    if dockerhub_tag is not None:
        return f"docker.io/jefzda/sweap-images:{dockerhub_tag}"

    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_docker_image_name_candidates(image_name: str) -> list[str]:
    """Return local image names to check, accounting for optional docker.io prefix."""
    candidates = [image_name]
    docker_io_prefix = "docker.io/"
    if image_name.startswith(docker_io_prefix):
        candidates.append(image_name[len(docker_io_prefix) :])
    else:
        candidates.append(f"{docker_io_prefix}{image_name}")
    return list(dict.fromkeys(candidates))


def local_docker_image_exists(image_name: str, *, executable: str = "docker") -> bool:
    """Check whether a Docker image exists locally without pulling it."""
    for candidate in get_docker_image_name_candidates(image_name):
        try:
            result = subprocess.run(
                [executable, "image", "inspect", candidate],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        if result.returncode == 0:
            return True
    return False


def filter_instances_with_local_images(
    instances: list[dict], *, docker_executable: str = "docker"
) -> list[dict]:
    """Keep only instances whose SWE-bench Docker image is available locally."""
    image_exists_cache: dict[str, bool] = {}
    filtered_instances = []
    skipped_instances = []

    for instance in instances:
        image_name = get_swebench_docker_image_name(instance)
        if image_name not in image_exists_cache:
            image_exists_cache[image_name] = local_docker_image_exists(image_name, executable=docker_executable)
        if image_exists_cache[image_name]:
            filtered_instances.append(instance)
        else:
            skipped_instances.append((instance["instance_id"], image_name))

    if skipped_instances:
        logger.info(
            f"Skipping {len(skipped_instances)} instances without local SWE-bench Docker images "
            f"({len(instances)} -> {len(filtered_instances)})"
        )
        for instance_id, image_name in skipped_instances:
            logger.debug(f"Skipping instance {instance_id}: local image not found for {image_name}")

    return filtered_instances


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = "docker://" + image_name

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def replace_api_base_port(api_base: str, port: int) -> str:
    """Return api_base with its host preserved and TCP port replaced."""
    parsed = urlsplit(api_base)
    host = parsed.hostname or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        netloc = f"{auth}@{netloc}"
    return urlunsplit((parsed.scheme or "http", netloc, parsed.path or "/v1", parsed.query, parsed.fragment))


def load_original_patches(original_patch_dir: Path) -> tuple[dict[str, str], set[str]]:
    """Load successful instance patches from an agent output directory."""
    preds_path = original_patch_dir / "preds.json"
    resolved_ids_path = original_patch_dir / "resolved_ids.json"
    if not preds_path.exists() or not resolved_ids_path.exists():
        raise FileNotFoundError(
            f"--original-patch must point to a directory containing preds.json and resolved_ids.json: "
            f"{original_patch_dir}"
        )
    preds = json.loads(preds_path.read_text())
    resolved_ids = set(json.loads(resolved_ids_path.read_text())["resolved_ids"])
    original_patches = {instance_id: prediction.get("model_patch", "") for instance_id, prediction in preds.items()}
    return original_patches, resolved_ids


def attach_original_patches(instances: list[dict], original_patch_dir: Path | None) -> list[dict]:
    if original_patch_dir is None:
        return instances

    original_patches, resolved_ids = load_original_patches(original_patch_dir)
    original_patches = {
        instance_id: patch for instance_id, patch in original_patches.items() if instance_id in resolved_ids
    }

    filtered_instances = []
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id in original_patches:
            filtered_instances.append({**instance, "original_patch": original_patches[instance_id]})
    logger.info(
        f"Filtered to {len(filtered_instances)} instances with resolved original patches "
        f"from {len(instances)} candidates"
    )
    return filtered_instances


def process_instance(
    instance: dict,
    output_dir: Path,
    traj_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = traj_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        info = agent.run(task, original_patch=instance.get("original_patch", ""))
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: Path = typer.Option(DEFAULT_OUTPUT_DIR, "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    port: int | None = typer.Option(None, "--port", help="Replace model.model_kwargs.api_base port", rich_help_panel="Basic"),
    original_patch: Path | None = typer.Option(None, "--original-patch", help="Path to an output directory containing preds.json and resolved_ids.json", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = output
    traj_path = output_path / "trajs"
    output_path.mkdir(parents=True, exist_ok=True)
    traj_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    logger.info(f"Trajectories will be saved to {traj_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    instances = attach_original_patches(instances, original_patch)

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)
    if port is not None:
        model_kwargs = config.setdefault("model", {}).setdefault("model_kwargs", {})
        model_kwargs["api_base"] = replace_api_base_port(model_kwargs.get("api_base", "http://localhost/v1"), port)

    docker_executable = config.get("environment", {}).get("executable", os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker"))
    instances = filter_instances_with_local_images(instances, docker_executable=docker_executable)
    logger.info(f"Running on {len(instances)} instances...")

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, traj_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
