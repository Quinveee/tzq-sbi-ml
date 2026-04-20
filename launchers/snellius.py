"""Snellius Slurm launcher."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from shlex import join as shell_join
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_cfg_file(cfg: DictConfig) -> Path:
    """Persist the resolved run config in a shared directory for Slurm jobs."""
    input_dir = PROJECT_ROOT / "inputs"
    input_dir.mkdir(exist_ok=True, parents=True)

    with tempfile.NamedTemporaryFile(
        mode="w", dir=input_dir, prefix="snellius_", suffix=".yaml", delete=False
    ) as tmp:
        tmp.write(OmegaConf.to_yaml(cfg))
        return Path(tmp.name).resolve()


def _resolve_path(path: str) -> Path:
    """Resolve relative paths from repository root."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (PROJECT_ROOT / path_obj).resolve()


def _serialize_export(
    *,
    export_all_env: bool,
    env: DictConfig | Mapping[str, object] | None,
) -> str | None:
    """Build a valid value for ``sbatch --export``."""
    env_map: Mapping[str, object]
    if env is None:
        env_map = {}
    elif isinstance(env, Mapping):
        env_map = env
    else:
        resolved = OmegaConf.to_container(env, resolve=True)
        if not isinstance(resolved, Mapping):
            raise TypeError("launcher.env must be a key-value mapping")
        env_map = resolved

    serialized = ",".join(f"{key}={value}" for key, value in env_map.items())

    if export_all_env:
        return f"ALL,{serialized}" if serialized else "ALL"
    return serialized or None


def _ensure_dataset_path(cfg: DictConfig) -> None:
    """
    Validate dataset path before submission and apply a Snellius fallback.

    If the configured dataset base does not exist (e.g. legacy `/data/...`),
    attempt `<repo>/data/<basename>`.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Resolved configuration has unexpected structure")

    dataset = resolved.get("dataset", {})
    if not isinstance(dataset, Mapping):
        return

    dataset_path_raw = dataset.get("path")
    if dataset_path_raw is None:
        return

    dataset_path = Path(str(dataset_path_raw))
    if dataset_path.exists():
        return

    base_raw = dataset.get("base")
    if base_raw is not None:
        fallback_base = (PROJECT_ROOT / "data" / Path(str(base_raw)).name).resolve()
        if fallback_base.exists():
            cfg.dataset.base = str(fallback_base)

            fixed = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(fixed, Mapping):
                fixed_dataset = fixed.get("dataset", {})
                if isinstance(fixed_dataset, Mapping):
                    fixed_path_raw = fixed_dataset.get("path")
                    if fixed_path_raw is not None and Path(str(fixed_path_raw)).exists():
                        print(
                            "Snellius launcher: dataset base not found; "
                            f"falling back to {fallback_base}"
                        )
                        return

    base_hint = str(base_raw) if base_raw is not None else "<unset>"
    raise FileNotFoundError(
        "Dataset path does not exist for Snellius run. "
        f"Resolved dataset.path={dataset_path}. "
        f"Resolved dataset.base={base_hint}. "
        "Please override dataset.base or dataset.path to a valid shared filesystem path."
    )


def launch(
    *,
    cfg: DictConfig,
    script: str,
    job_name: str,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    time: str = "04:00:00",
    cpus_per_task: int = 4,
    mem: str = "32G",
    gpus: int = 0,
    constraint: str | None = None,
    output: str = "logs/slurm-%j.out",
    error: str = "logs/slurm-%j.err",
    chdir: str | None = ".",
    export_all_env: bool = True,
    env: DictConfig | Mapping[str, object] | None = None,
    extra_args: Iterable[str] | None = None,
    **_,
) -> None:
    """
    Submit one Slurm job that executes the worker with a serialized Hydra config.

    This launcher does not stage datasets to compute nodes. It assumes the shared
    filesystem used on Snellius so the worker can read paths directly.
    """
    if shutil.which("sbatch") is None:
        raise RuntimeError("Could not find 'sbatch' in PATH.")

    script_path = _resolve_path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"Slurm script not found: {script_path}")

    output_path = _resolve_path(output)
    error_path = _resolve_path(error)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    error_path.parent.mkdir(exist_ok=True, parents=True)

    _ensure_dataset_path(cfg)

    chdir_path = _resolve_path(chdir) if chdir is not None else None

    cfg_path = _write_cfg_file(cfg)

    command = ["sbatch"]

    def _add(flag: str, value: object | None) -> None:
        if value is not None and value != "":
            command.extend([flag, str(value)])

    _add("--job-name", job_name)
    _add("--partition", partition)
    _add("--account", account)
    _add("--qos", qos)
    _add("--time", time)
    _add("--cpus-per-task", cpus_per_task)
    _add("--mem", mem)
    if gpus and int(gpus) > 0:
        _add("--gpus", int(gpus))
    _add("--constraint", constraint)
    _add("--output", output_path)
    _add("--error", error_path)
    _add("--chdir", chdir_path)

    export_value = _serialize_export(export_all_env=export_all_env, env=env)
    if export_value:
        _add("--export", export_value)

    if extra_args:
        command.extend(str(arg) for arg in extra_args)

    command.extend([str(script_path), str(cfg_path)])

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "No output from sbatch"
        raise RuntimeError(
            "Slurm submission failed. "
            f"sbatch exit code={result.returncode}. "
            f"Command: {shell_join(command)}\n"
            f"sbatch output:\n{details}"
        )

    if result.stdout.strip():
        print(result.stdout.strip())
    elif result.stderr.strip():
        print(result.stderr.strip())
