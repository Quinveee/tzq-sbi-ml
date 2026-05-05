"""Weights & Biases sweep orchestration experiment.

Creates one wandb sweep per model preset (mlp, transformer, lloca, ...),
sharing the same Bayesian HPO search space. A pool of N worker processes
pulls (preset, sweep_id) jobs from a shared queue and runs `wandb.agent`
with count=1 for each — that gives round-robin across presets with dynamic
load balancing. Workers share the GPU through a CUDA MPS daemon.

Set ``exp.sweep.parallel.enabled=false`` to fall back to a single in-process
agent that drains the queue sequentially.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping as MappingABC
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from helpers.derive_config import derive_config, load_conf_from

from .base.base_experiment import BaseExperiment
from .logger import LOGGER as _LOGGER

if TYPE_CHECKING:
    pass

LOGGER = _LOGGER.getChild(__name__)


# ---------------------------------------------------------------------------
# CUDA MPS / GPU helpers (single GPU per worker — mirrors utils/multi_tune.py)
# ---------------------------------------------------------------------------

def _gpu_total_gb(device: int = 0) -> float:
    try:
        import torch  # local import: keeps parent import cheap when no GPU
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(device).total_memory / 1e9
    except Exception:
        return 0.0


def _gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_mps(pipe_dir: str, log_dir: str, device_id: int = 0) -> bool:
    """Launch a CUDA MPS control daemon scoped to a single GPU."""
    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(device_id),
        "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
        "CUDA_MPS_LOG_DIRECTORY": log_dir,
    }
    try:
        r = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True
        )
    except FileNotFoundError:
        return False
    return r.returncode == 0


def _stop_mps(pipe_dir: str) -> None:
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe_dir}
    subprocess.run(
        ["sh", "-c", "echo quit | nvidia-cuda-mps-control"],
        env=env,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# OmegaConf resolver registration (re-run inside spawned workers + probe)
# ---------------------------------------------------------------------------

def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env",
            lambda key: {
                "prefix": Path(sys.executable).parent,
                "cwd": os.getcwd(),
            }.get(key),
        )
    if not OmegaConf.has_resolver("sum"):
        OmegaConf.register_new_resolver("sum", lambda *values: sum(values))
    if not OmegaConf.has_resolver("prod"):
        OmegaConf.register_new_resolver(
            "prod", lambda *values: math.prod(int(v) for v in values)
        )


# ---------------------------------------------------------------------------
# Probe subprocess: runs one short trial and prints peak GPU memory in GB
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = '''
import os, sys, math
from pathlib import Path

from omegaconf import OmegaConf
from hydra.utils import instantiate

if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver(
        "env",
        lambda key: {"prefix": Path(sys.executable).parent, "cwd": os.getcwd()}.get(key),
    )
if not OmegaConf.has_resolver("sum"):
    OmegaConf.register_new_resolver("sum", lambda *v: sum(v))
if not OmegaConf.has_resolver("prod"):
    OmegaConf.register_new_resolver("prod", lambda *v: math.prod(int(x) for x in v))

import torch

cfg_path = sys.argv[1]
cfg = OmegaConf.load(cfg_path)
OmegaConf.set_struct(cfg, False)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
try:
    instantiate(cfg.launcher)(cfg=cfg)
except Exception as exc:
    print(f"PROBE_ERROR={exc!r}", flush=True)
peak_gb = (
    torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
)
print(f"PROBE_PEAK_GB={peak_gb:.4f}", flush=True)
'''


# ---------------------------------------------------------------------------
# Worker entry point (top-level so it survives multiprocessing 'spawn')
# ---------------------------------------------------------------------------

def _sweep_worker_main(
    worker_id: int,
    cfg_yaml_path: str,
    project: str,
    entity: Optional[str],
    n_gpus: int,
    mps_pipe_dirs: dict[int, str],
    job_queue: "mp.queues.Queue",
    log_level: int,
) -> None:
    """Pull (preset, sweep_id) jobs from the queue and run one trial each."""
    gpu_id = worker_id % max(n_gpus, 1)
    if n_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if mps_pipe_dirs and gpu_id in mps_pipe_dirs:
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dirs[gpu_id]

    worker_tmp = tempfile.mkdtemp(prefix=f"tzq_sweep_worker_{worker_id}_")
    os.environ["TMPDIR"] = worker_tmp
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(_find_free_port())

    # Stagger startup so workers don't all hit wandb / GPU init in lockstep.
    time.sleep(worker_id * 0.5)

    fmt = f"%(asctime)s  W{worker_id:<2}  %(name)-24s  %(levelname)s  %(message)s"
    logging.basicConfig(level=log_level, format=fmt, datefmt="%H:%M:%S", force=True)
    wlog = logging.getLogger(f"sweep.worker.{worker_id}")

    _register_resolvers()
    cfg = OmegaConf.load(cfg_yaml_path)
    OmegaConf.set_struct(cfg, False)

    from experiments.sweep import ExperimentSweep  # avoid stale import in spawn
    exp = ExperimentSweep(cfg=cfg, key="sweep")

    wlog.info(
        f"Worker {worker_id} ready (gpu={gpu_id}, mps={'on' if mps_pipe_dirs else 'off'})"
    )

    try:
        while True:
            try:
                item = job_queue.get(timeout=1.0)
            except Exception:
                continue
            if item is None:
                break
            preset_name, sweep_id = item
            runner = partial(exp._run_single_trial_for_preset, preset_name)
            try:
                wandb.agent(
                    sweep_id=sweep_id,
                    function=runner,
                    project=project,
                    entity=entity,
                    count=1,
                )
            except Exception as exc:
                wlog.warning(
                    f"trial failed for preset={preset_name} sweep={sweep_id}: {exc}"
                )
    finally:
        try:
            import shutil
            shutil.rmtree(worker_tmp, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ExperimentSweep
# ---------------------------------------------------------------------------

class ExperimentSweep(BaseExperiment):
    """Run a wandb HPO sweep per model preset, executing trials in parallel."""

    def __init__(self, *args, sweep: Any = None, **kwds) -> None:
        # `sweep` is passed from cfg.exp.sweep by hydra's instantiate call.
        super().__init__(*args, **kwds)
        self.sweep_cfg = sweep
        if self.sweep_cfg is None and "sweep" in self.cfg:
            self.sweep_cfg = self.cfg.sweep
        if self.sweep_cfg is None and "sweep" in self.cfg.exp:
            self.sweep_cfg = self.cfg.exp.sweep
        if self.sweep_cfg is None:
            raise ValueError("Sweep configuration not found. Expected exp.sweep settings.")

    def _init(self) -> None:
        # BaseExperiment handles run-dir/checkpoints setup.
        return

    @staticmethod
    def _apply_updates(cfg: DictConfig, updates: Mapping[str, Any]) -> None:
        """Apply dotted-path updates into an OmegaConf config."""
        for key, value in updates.items():
            OmegaConf.update(cfg, key, value, merge=False, force_add=True)

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        """Coerce common serialized scalar values from sweep configs."""
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower == "true":
                return True
            if lower == "false":
                return False
        return value

    @classmethod
    def _flatten_updates(
        cls,
        updates: Mapping[str, Any],
        prefix: str = "",
    ) -> dict[str, Any]:
        """Flatten nested dictionaries to dotted-path updates."""
        flat: dict[str, Any] = {}
        for key, value in updates.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, MappingABC):
                flat.update(cls._flatten_updates(value, full_key))
            else:
                flat[full_key] = cls._coerce_value(value)
        return flat

    @staticmethod
    def _pop_first(
        updates: dict[str, Any],
        keys: tuple[str, ...],
        default: Any,
    ) -> Any:
        """Pop and return the first present key from a set of aliases."""
        for key in keys:
            if key in updates:
                return updates.pop(key)
        return default

    def _expand_preset(self, sampled_updates: dict[str, Any]) -> dict[str, Any]:
        """Expand a sampled ``_preset`` value into concrete config overrides."""
        preset_name = sampled_updates.pop("_preset", None)
        if preset_name is None:
            return sampled_updates
        presets = self.sweep_cfg.get("presets", None)
        if presets is None or preset_name not in presets:
            raise ValueError(
                f"sweep _preset='{preset_name}' has no entry under exp.sweep.presets"
            )
        preset_overrides = OmegaConf.to_container(presets[preset_name], resolve=True) or {}
        flat_overrides = self._flatten_updates(preset_overrides)
        for key, value in flat_overrides.items():
            sampled_updates.setdefault(key, value)
        return sampled_updates

    def _resolve_trial_target(
        self,
        sampled_updates: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Resolve per-trial target config groups from sampled values."""
        updates = sampled_updates.copy()
        base_target = self.sweep_cfg.target

        exp_key = self._pop_first(
            updates,
            ("sweep.target.exp", "target.exp", "exp.key", "exp"),
            base_target.get("exp", None),
        )
        model_key = self._pop_first(
            updates,
            ("sweep.target.model", "target.model", "model.key", "model"),
            base_target.get("model", None),
        )
        dataset_key = self._pop_first(
            updates,
            (
                "sweep.target.dataset",
                "target.dataset",
                "dataset.key",
                "dataset",
            ),
            base_target.get("dataset", None),
        )
        launcher_key = self._pop_first(
            updates,
            (
                "sweep.target.launcher",
                "target.launcher",
                "launcher.key",
                "launcher",
            ),
            base_target.get("launcher", "local"),
        )

        for ignored in (
            "data.run_dir",
            "data.run_dir_base",
            "dataset.path",
            "dataset.events_file",
            "program",
        ):
            updates.pop(ignored, None)

        updates = {
            key: value
            for key, value in updates.items()
            if not any(part.startswith("_") for part in key.split("."))
        }

        target = {
            "exp": str(exp_key) if exp_key is not None else "",
            "model": str(model_key) if model_key is not None else "",
            "dataset": str(dataset_key) if dataset_key is not None else "",
            "launcher": str(launcher_key) if launcher_key is not None else "local",
        }
        return target, updates

    def _build_trial_cfg(self, sampled: Mapping[str, Any]) -> DictConfig:
        """Build one concrete trial configuration from sweep samples."""
        trial_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
        OmegaConf.set_struct(trial_cfg, False)

        flat_sampled = self._flatten_updates(sampled)
        flat_sampled = self._expand_preset(flat_sampled)
        target, sampled_updates = self._resolve_trial_target(flat_sampled)

        for field in ("exp", "model", "dataset"):
            if not target[field]:
                raise ValueError(f"sweep.target.{field} is required")
        if target["exp"] == "sweep":
            raise ValueError("sweep.target.exp cannot be 'sweep'")

        trial_cfg.merge_with(
            load_conf_from(Path("conf/exp") / target["exp"], merge_on="exp")
        )
        trial_cfg.merge_with(
            load_conf_from(Path("conf/model") / target["model"], merge_on="model")
        )
        trial_cfg.merge_with(
            load_conf_from(Path("conf/dataset") / target["dataset"], merge_on="dataset")
        )
        launcher_key = target["launcher"]
        trial_cfg.merge_with(
            load_conf_from(Path("conf/launcher") / launcher_key, merge_on="launcher")
        )
        if launcher_key != "local":
            raise ValueError(
                f"exp.sweep.target.launcher='{launcher_key}' is not supported. "
                "Trials must run in-process (target.launcher=local) so the "
                "wandb agent can attach wandb.init, capture metrics during "
                "training, and call run.finish() synchronously. The htcondor "
                "launcher is fire-and-forget — using it for trials would make "
                "every trial look crashed to the bayesian sampler.\n\n"
                "To run the SWEEP itself on the cluster, leave "
                "target.launcher=local and pass the cluster launcher at the "
                "top level instead, e.g.:\n"
                "    python main.py exp=sweep launcher=htcondor launcher.job_category=long\n"
                "The whole sweep (orchestrator + MPS workers + all trials) "
                "then runs on a single GPU node."
            )

        fixed_updates = OmegaConf.to_container(
            self.sweep_cfg.get("fixed", {}),
            resolve=True,
        )
        fixed_flat: dict[str, Any] = {}
        if isinstance(fixed_updates, MappingABC):
            fixed_flat = self._flatten_updates(fixed_updates)
            self._apply_updates(trial_cfg, fixed_flat)
        self._apply_updates(trial_cfg, sampled_updates)

        if "sweep" in trial_cfg.exp:
            del trial_cfg.exp.sweep

        trial_cfg.modes.wandb = True
        if wandb.run is not None:
            trial_cfg.data.run = wandb.run.id

        derived = derive_config(trial_cfg)

        # derive_config auto-loads conf/_auto/{loss,dataset,limits}/... which
        # can clobber sampled overrides like loss.log_r_clip. Re-apply user
        # updates so the search space wins.
        OmegaConf.set_struct(derived, False)
        if fixed_flat:
            self._apply_updates(derived, fixed_flat)
        self._apply_updates(derived, sampled_updates)
        OmegaConf.set_struct(derived, True)

        return derived

    def _run_single_trial_for_preset(self, preset_name: str) -> None:
        """wandb agent callback executing one trial for a given preset."""
        sweep_cfg = self.sweep_cfg
        tags = list(sweep_cfg.get("tags", [])) + [f"preset:{preset_name}"]
        run = wandb.init(
            project=sweep_cfg.project,
            entity=sweep_cfg.get("entity", None),
            dir="runs/",
            job_type="sweep-trial",
            tags=tags,
        )
        try:
            sampled = dict(wandb.config)
            sampled["_preset"] = preset_name
            trial_cfg = self._build_trial_cfg(sampled)
            instantiate(trial_cfg.launcher)(cfg=trial_cfg)
        finally:
            if run is not None:
                run.finish()

    # -----------------------------------------------------------------------
    # Per-preset wandb sweep creation
    # -----------------------------------------------------------------------

    def _create_per_preset_sweeps(self) -> dict[str, str]:
        """Create one wandb sweep per preset and return {preset: sweep_id}.

        Sweep names embed dataset/exp/preset so multiple experiment runs
        (1d/3d × ratio/score × 5 presets) don't collide visually in wandb.

        If ``exp.sweep.sweep_ids`` is provided (preset → id), reuse those ids.
        Also writes the {preset → sweep_id} map to ``data.run_dir/sweep_ids.json``
        so the aggregation script can find them.
        """
        sweep_cfg = self.sweep_cfg
        existing = sweep_cfg.get("sweep_ids", None)
        if existing:
            ids = OmegaConf.to_container(existing, resolve=True) or {}
            LOGGER.info(f"Reusing pre-existing wandb sweep ids: {ids}")
            sweep_ids = {str(k): str(v) for k, v in ids.items()}
        else:
            base_spec = OmegaConf.to_container(sweep_cfg.spec, resolve=True)
            base_name = str(base_spec.get("name", "hpo"))
            target = sweep_cfg.target
            scope = f"{target.dataset}-{target.exp}"
            sweep_ids = {}
            for preset_name in sweep_cfg.presets:
                spec = dict(base_spec)
                spec["name"] = f"{base_name}-{scope}-{preset_name}"
                sweep_id = wandb.sweep(
                    sweep=spec,
                    project=sweep_cfg.project,
                    entity=sweep_cfg.get("entity", None),
                )
                LOGGER.info(
                    f"Created wandb sweep {spec['name']} preset={preset_name} id={sweep_id}"
                )
                sweep_ids[str(preset_name)] = sweep_id

        # Persist for the aggregation script.
        try:
            ids_path = Path(self.cfg.data.run_dir) / "sweep_ids.json"
            ids_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            payload = {
                "project": str(sweep_cfg.project),
                "entity": sweep_cfg.get("entity", None),
                "dataset": str(sweep_cfg.target.dataset),
                "exp": str(sweep_cfg.target.exp),
                "sweep_ids": sweep_ids,
            }
            ids_path.write_text(json.dumps(payload, indent=2))
            LOGGER.info(f"Wrote sweep id map to {ids_path}")
        except Exception as exc:
            LOGGER.warning(f"Could not persist sweep_ids.json: {exc}")
        return sweep_ids

    # -----------------------------------------------------------------------
    # Parallel orchestration
    # -----------------------------------------------------------------------

    def _build_probe_cfg(self) -> DictConfig:
        """Build a minimal trial cfg for the memory probe (1 epoch, clamped)."""
        probe_cfg = self.sweep_cfg.parallel.probe
        preset = probe_cfg.get("preset", None)
        if preset is None:
            preset = next(iter(self.sweep_cfg.presets))
        sampled = {
            "_preset": preset,
            "train.epochs": 1,
            "train.clamp_samples": 512,
            "train.lr_warmup": 0,
            "modes.train": True,
            "modes.eval": False,
            "modes.plot": False,
        }
        cfg = self._build_trial_cfg(sampled)
        OmegaConf.set_struct(cfg, False)
        cfg.modes.wandb = False
        return cfg

    def _probe_memory_gb(self) -> Optional[float]:
        """Spawn a subprocess that runs a brief trial and reports peak GPU memory."""
        if _gpu_count() == 0:
            return None
        probe_cfg = self.sweep_cfg.parallel.probe
        timeout_s = int(probe_cfg.get("timeout_s", 600))

        try:
            cfg = self._build_probe_cfg()
        except Exception as exc:
            LOGGER.warning(f"Could not build probe config: {exc}")
            return None

        with tempfile.TemporaryDirectory(prefix="tzq_sweep_probe_") as tmpdir:
            cfg_path = os.path.join(tmpdir, "probe_cfg.yaml")
            OmegaConf.save(cfg, cfg_path, resolve=False)

            script_path = os.path.join(tmpdir, "_probe_script.py")
            with open(script_path, "w") as fh:
                fh.write(_PROBE_SCRIPT)

            try:
                LOGGER.info(
                    f"Probing peak memory: preset={probe_cfg.get('preset', '?')} "
                    f"timeout={timeout_s}s"
                )
                r = subprocess.run(
                    [sys.executable, script_path, cfg_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                LOGGER.warning("Probe timed out")
                return None
            except Exception as exc:
                LOGGER.warning(f"Probe subprocess failed: {exc}")
                return None

        for line in reversed(r.stdout.splitlines()):
            if line.startswith("PROBE_PEAK_GB="):
                try:
                    return float(line.split("=", 1)[1])
                except ValueError:
                    pass
        if r.returncode != 0 and r.stderr:
            LOGGER.debug("Probe stderr tail:\n" + "\n".join(r.stderr.splitlines()[-15:]))
        return None

    def _decide_n_parallel(self) -> tuple[int, dict[int, str]]:
        """Pick worker count and return (n_parallel, mps_pipe_dirs)."""
        par_cfg = self.sweep_cfg.parallel
        n_gpus = _gpu_count()
        max_workers = int(par_cfg.get("max", 8))
        safety = float(par_cfg.get("safety_margin", 0.80))

        n_override = par_cfg.get("n_override", None)
        if n_override is not None:
            n_parallel = max(1, int(n_override))
            LOGGER.info(f"Using n_parallel override: {n_parallel}")
        elif n_gpus == 0:
            n_parallel = 1
            LOGGER.info("No CUDA GPU detected — running 1 worker on CPU")
        else:
            probe_enabled = bool(par_cfg.probe.get("enabled", True))
            per_trial_gb = self._probe_memory_gb() if probe_enabled else None
            if per_trial_gb is None:
                per_trial_gb = float(par_cfg.probe.get("fallback_gb", 4.0))
                LOGGER.warning(f"Using fallback per-trial memory = {per_trial_gb:.2f} GB")
            total_gb = _gpu_total_gb()
            slot_gb = per_trial_gb * 1.3 + 0.4
            n_per_gpu = max(1, int((total_gb * safety) / slot_gb))
            n_parallel = min(n_per_gpu * n_gpus, max_workers)
            LOGGER.info(
                f"GPU total={total_gb:.1f} GB  per-trial={per_trial_gb:.2f} GB  "
                f"slot={slot_gb:.2f} GB  → n_parallel={n_parallel} "
                f"(n_per_gpu={n_per_gpu}, n_gpus={n_gpus}, cap={max_workers})"
            )

        mps_pipe_dirs: dict[int, str] = {}
        if (
            n_gpus > 0
            and bool(par_cfg.get("mps", True))
            and n_parallel > 1
        ):
            pid = os.getpid()
            for dev in range(n_gpus):
                pipe_dir = f"/tmp/tzq_sweep_mps_pipe_{pid}_{dev}"
                log_dir = f"/tmp/tzq_sweep_mps_log_{pid}_{dev}"
                if _start_mps(pipe_dir, log_dir, device_id=dev):
                    mps_pipe_dirs[dev] = pipe_dir
                    LOGGER.info(f"CUDA MPS started for GPU {dev} (pipe={pipe_dir})")
                else:
                    LOGGER.warning(f"Could not start CUDA MPS for GPU {dev}; running without it")

        return n_parallel, mps_pipe_dirs

    def _build_job_queue(
        self,
        ctx: mp.context.BaseContext,
        sweep_ids: dict[str, str],
        n_workers: int,
    ) -> "mp.queues.Queue":
        """Pre-fill a shared queue with (preset, sweep_id) jobs in round-robin order."""
        queue = ctx.Queue()
        n_per_model = int(self.sweep_cfg.get("n_trials_per_model", 30))
        # Round-robin: trial_idx outer, preset inner — interleaves presets so
        # bayesian state gets distributed feedback early.
        for _trial_idx in range(n_per_model):
            for preset_name, sweep_id in sweep_ids.items():
                queue.put((preset_name, sweep_id))
        for _ in range(n_workers):
            queue.put(None)  # sentinel per worker
        LOGGER.info(
            f"Queued {n_per_model * len(sweep_ids)} trials across {len(sweep_ids)} presets"
        )
        return queue

    def _run_parallel(self, sweep_ids: dict[str, str]) -> None:
        """Spawn N workers, each draining the shared (preset, sweep_id) queue."""
        n_parallel, mps_pipe_dirs = self._decide_n_parallel()
        n_gpus = _gpu_count()
        sweep_cfg = self.sweep_cfg
        project = sweep_cfg.project
        entity = sweep_cfg.get("entity", None)

        ctx = mp.get_context("spawn")
        job_queue = self._build_job_queue(ctx, sweep_ids, n_parallel)

        with tempfile.TemporaryDirectory(prefix="tzq_sweep_") as tmpdir:
            cfg_yaml_path = os.path.join(tmpdir, "sweep_cfg.yaml")
            OmegaConf.save(self.cfg, cfg_yaml_path, resolve=False)

            try:
                if n_parallel == 1:
                    LOGGER.info("Running a single in-process agent over all presets")
                    while True:
                        try:
                            item = job_queue.get(timeout=1.0)
                        except Exception:
                            continue
                        if item is None:
                            break
                        preset_name, sweep_id = item
                        runner = partial(self._run_single_trial_for_preset, preset_name)
                        try:
                            wandb.agent(
                                sweep_id=sweep_id,
                                function=runner,
                                project=project,
                                entity=entity,
                                count=1,
                            )
                        except Exception as exc:
                            LOGGER.warning(
                                f"trial failed for preset={preset_name} "
                                f"sweep={sweep_id}: {exc}"
                            )
                    return

                procs: list[mp.process.BaseProcess] = []
                for w in range(n_parallel):
                    p = ctx.Process(
                        target=_sweep_worker_main,
                        kwargs=dict(
                            worker_id=w,
                            cfg_yaml_path=cfg_yaml_path,
                            project=project,
                            entity=entity,
                            n_gpus=max(n_gpus, 1),
                            mps_pipe_dirs=dict(mps_pipe_dirs),
                            job_queue=job_queue,
                            log_level=logging.INFO,
                        ),
                        name=f"sweep-worker-{w}",
                    )
                    p.start()
                    procs.append(p)

                LOGGER.info(
                    f"Spawned {n_parallel} worker(s) "
                    f"(presets={list(sweep_ids.keys())})"
                )

                exit_codes = []
                for p in procs:
                    p.join()
                    exit_codes.append(p.exitcode)
                if any(code not in (0, None) for code in exit_codes):
                    LOGGER.warning(f"Worker exit codes: {exit_codes}")
            finally:
                for dev, pipe_dir in mps_pipe_dirs.items():
                    _stop_mps(pipe_dir)
                    LOGGER.info(f"CUDA MPS stopped for GPU {dev}")

    def _run(self) -> None:
        """Create one sweep per preset and drain trials via parallel agents."""
        wandb.login()
        sweep_ids = self._create_per_preset_sweeps()
        if not sweep_ids:
            raise ValueError("No presets defined under exp.sweep.presets")
        self._run_parallel(sweep_ids)
