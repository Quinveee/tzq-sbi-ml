"""Weights & Biases sweep orchestration experiment."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from helpers.derive_config import derive_config, load_conf_from

from .base.base_experiment import BaseExperiment
from .logger import LOGGER as _LOGGER

if TYPE_CHECKING:
    from omegaconf import DictConfig

LOGGER = _LOGGER.getChild(__name__)


class ExperimentSweep(BaseExperiment):
    """Run a wandb parameter sweep over one target experiment setup."""

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
        target, sampled_updates = self._resolve_trial_target(flat_sampled)

        for field in ("exp", "model", "dataset"):
            if not target[field]:
                raise ValueError(f"sweep.target.{field} is required")
        if target["exp"] == "sweep":
            raise ValueError("sweep.target.exp cannot be 'sweep'")

        # Re-compose config groups from the selected target setup.
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
                "sweep.target.launcher must be 'local' so sweep trials execute in the wandb agent process"
            )

        # Static overrides plus sampled hyperparameters.
        fixed_updates = OmegaConf.to_container(
            self.sweep_cfg.get("fixed", {}),
            resolve=True,
        )
        if isinstance(fixed_updates, MappingABC):
            self._apply_updates(trial_cfg, self._flatten_updates(fixed_updates))
        self._apply_updates(trial_cfg, sampled_updates)

        # Remove sweep-only payload from exp before instantiating target experiments.
        if "sweep" in trial_cfg.exp:
            del trial_cfg.exp.sweep

        # Force wandb logging and use run id to isolate output directories.
        trial_cfg.modes.wandb = True
        if wandb.run is not None:
            trial_cfg.data.run = wandb.run.id

        return derive_config(trial_cfg)

    def _run_single_trial(self) -> None:
        """wandb agent callback that executes one trial."""
        sweep_cfg = self.sweep_cfg
        run = wandb.init(
            project=sweep_cfg.project,
            entity=sweep_cfg.get("entity", None),
            dir="runs/",
            job_type="sweep-trial",
            tags=list(sweep_cfg.get("tags", [])),
        )

        try:
            sampled = dict(wandb.config)
            trial_cfg = self._build_trial_cfg(sampled)
            instantiate(trial_cfg.launcher)(cfg=trial_cfg)
        finally:
            if run is not None:
                run.finish()

    def _run(self) -> None:
        """Create (or reuse) a sweep and execute trials via wandb agent."""
        sweep_cfg = self.sweep_cfg
        entity = sweep_cfg.get("entity", None)
        sweep_id = sweep_cfg.get("id", None)

        wandb.login()

        if sweep_id:
            LOGGER.info("Reusing existing wandb sweep id=%s", sweep_id)
        else:
            sweep_spec = OmegaConf.to_container(sweep_cfg.spec, resolve=True)
            sweep_id = wandb.sweep(
                sweep=sweep_spec,
                project=sweep_cfg.project,
                entity=entity,
            )
            LOGGER.info("Created wandb sweep id=%s", sweep_id)

        count = sweep_cfg.get("count", None)
        count = int(count) if count is not None else None
        if count is None:
            wandb.agent(
                sweep_id=sweep_id,
                function=self._run_single_trial,
                project=sweep_cfg.project,
                entity=entity,
            )
        else:
            wandb.agent(
                sweep_id=sweep_id,
                function=self._run_single_trial,
                count=count,
                project=sweep_cfg.project,
                entity=entity,
            )
