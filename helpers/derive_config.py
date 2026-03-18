"""
Function to dynamically derive configuration fields based on
CL-provided configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def load_conf_from(path: Path, merge_on: Optional[str] = None) -> DictConfig:
    """
    Helper function to load configuration objects dynamically from
    files

    :param path: Path of the configuration .yaml file to load
    :type path: Path
    :param merge_on: Optionally merge on this key (nested dict)
    :type merge_on: Optional[str]
    :return: Loaded configuration object
    :rtype: DictConfig
    """
    # Check for the existence of base configuration files on the same folder
    # of the target configuration file
    base_path = path.parent / "_base.yaml"
    base = OmegaConf.load(base_path) if base_path.exists() else {}

    # Load main configuration file
    cfg = OmegaConf.merge(base, OmegaConf.load(path.with_suffix(".yaml")))

    # Optinally nest the configuration file on a custom key
    if merge_on is not None:
        return OmegaConf.create({merge_on: cfg})

    return cfg


def derive_config(cfg: DictConfig) -> DictConfig:
    """
    Dynamically derive certain configuration fields based on user-provided config

    :param cfg: Parsed used-provided configuration object
    :type cfg: DictConfig
    :return: Original configuration object with derived additions
    :rtype: DictConfig
    """
    OmegaConf.set_struct(cfg, False)

    # Dynamically loaded configs dir
    auto_dir = Path("conf/_auto")

    # Limits configuration depend on dataset configuration
    cfg.merge_with(
        load_conf_from(auto_dir / "limits" / cfg.dataset.key, merge_on="limits")
    )

    # The specific experiment and model wrappers depend on the specified experiment
    # and the specified model
    # For example, experiment=local and model=lgatr
    # will correspond to ExperimentLocalParticles and LocalLGATrWrapper
    model_key = cfg.model.key if cfg.model.key else "noop"

    # Transformer can optionally consume feature-level data instead of particles.
    # We keep backwards compatibility by defaulting to particles when unset.
    transformer_input = "particles"
    if model_key == "transformer":
        transformer_input = cfg.model.get("input_level", "particles")
        assert transformer_input in (
            "particles",
            "features",
        ), f"Invalid transformer input_level={transformer_input}"
        # This flag is only used to derive auto-config choices and should not
        # be passed to the wrapper constructor.
        if "input_level" in cfg.model:
            del cfg.model.input_level

    exp_model_key = f"{cfg.exp.key}_{model_key}"
    if model_key == "transformer" and transformer_input == "features":
        exp_model_key = f"{exp_model_key}_features"

    cfg.merge_with(load_conf_from(auto_dir / "exp_model" / exp_model_key))

    # Load the right dataset corresponding to model type.
    # MLP and histos models use feature-level inputs while LGATr and
    # Transformer use particles by default.
    dataset_key = model_key
    if model_key == "transformer" and transformer_input == "features":
        dataset_key = "transformer_features"
    cfg.merge_with(load_conf_from(auto_dir / "dataset" / dataset_key, merge_on="dataset"))

    # We use fixed loss functions for either LLR regression or Score regression
    # You can change them changing the symlinks in `conf/_auto/loss`
    loss_path = (auto_dir / "loss" / cfg.exp.key).with_suffix(".yaml")
    if loss_path.exists():
        cfg.merge_with(load_conf_from(loss_path, merge_on="loss"))

    # Allow model yamls to override global train params by defining a `train:` block.
    # Model-specific values take precedence over the defaults in config.yaml.
    # The `train` key is then removed from cfg.model so it isn't passed as a
    # kwarg to the wrapper constructor during instantiation.
    if "train" in cfg.model:
        cfg.train = OmegaConf.merge(cfg.train, cfg.model.train)
        del cfg.model.train

    # Derive run_dir model segment based on optional Transformer LLoCa mode.
    # Keep default folder names unchanged for all other models.
    run_model_key = model_key
    if model_key == "transformer":
        lloca_cfg = cfg.model.get("LLoCa", {})
        if lloca_cfg.get("active", False):
            run_model_key = f"{model_key}_lloca"

    cfg.data.run_dir = (
        f"{cfg.data.run_dir_base}/{cfg.dataset.key}/{cfg.exp.key}/{run_model_key}/{cfg.data.run}"
    )

    OmegaConf.set_struct(cfg, True)

    return cfg
