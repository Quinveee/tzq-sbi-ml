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
    use_preprocessed = bool(cfg.data.get("preprocessed", False))
    n_preprocessed_features = int(cfg.data.get("n_preprocessed_features", 0) or 0)

    if use_preprocessed and n_preprocessed_features <= 0:
        raise ValueError(
            "When data.preprocessed=true, data.n_preprocessed_features must be > 0"
        )

    if use_preprocessed and model_key not in ("transformer", "lgatr", "cnf", "lorentznet"):
        raise ValueError(
            "data.preprocessed=true is only supported for model=transformer, lgatr, cnf or lorentznet"
        )

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

    # Optionally inject engineered, preprocessed event-level features.
    # Transformer consumes them as extra per-particle channels (same conditioning
    # style as theta), while LGATr consumes them as scalar channels.
    if use_preprocessed and model_key == "transformer":
        if transformer_input != "particles":
            raise ValueError(
                "data.preprocessed=true is only supported with model.input_level=particles"
            )

        # `dim_in` may come from an interpolation such as `${sum:...}`.
        # Resolve it before applying the preprocessed-feature increment.
        resolved_net = OmegaConf.to_container(cfg.model.net, resolve=True)
        dim_in = int(resolved_net["dim_in"])
        cfg.model.net.dim_in = dim_in + n_preprocessed_features

    # LGATr with parametrized (ratio / joint) experiments routes θ through the
    # scalar pathway — θ is Lorentz-invariant so it doesn't belong on the
    # multivector path, and using scalar channels avoids burning GA ops on
    # 15-of-16 zero slots. Local / CNF LGATr variants don't take θ as encoder
    # input, so they keep the default in_s_channels.
    if model_key == "lgatr" and cfg.exp.key in ("ratio", "joint"):
        theta_dim_val = int(cfg.dataset.theta_dim)
        cfg.model.net.in_s_channels = theta_dim_val
        cfg.model.net.out_s_channels = theta_dim_val

    if use_preprocessed and model_key == "lgatr":
        resolved_net = OmegaConf.to_container(cfg.model.net, resolve=True)
        in_s = int(resolved_net.get("in_s_channels", 0) or 0)
        out_s = int(resolved_net.get("out_s_channels", 0) or 0)
        cfg.model.net.in_s_channels = in_s + n_preprocessed_features
        cfg.model.net.out_s_channels = out_s + n_preprocessed_features

    # CNF uses LGATr as its encoder, so it shares the scalar-channel plumbing.
    if use_preprocessed and model_key == "cnf":
        cfg.model.net.in_s_channels = n_preprocessed_features
        # Keep out_s_channels as defined in conf/model/cnf.yaml (it contributes to token_dim).

    # LorentzNet consumes preprocessed features as extra per-particle scalar
    # channels, broadcast from event-level to particle-level in the wrapper.
    if use_preprocessed and model_key == "lorentznet":
        resolved_net = OmegaConf.to_container(cfg.model.net, resolve=True)
        n_scalar = int(resolved_net["n_scalar"])
        cfg.model.net.n_scalar = n_scalar + n_preprocessed_features

    # When MET features are present and preprocessed features are NOT used,
    # the raw MET (pt, phi) is fed directly to the model as 2 extra scalars.
    # (When preprocessed=true, MET-derived quantities are already part of the
    # high-level feature vector, so no extra adjustment is needed.)
    use_met = bool(cfg.data.get("met", False))
    n_met_features = 2  # MET pt and phi

    if use_met and not use_preprocessed and model_key == "transformer":
        if transformer_input != "particles":
            raise ValueError(
                "data.met=true is only supported with model.input_level=particles"
            )
        resolved_net = OmegaConf.to_container(cfg.model.net, resolve=True)
        dim_in = int(resolved_net["dim_in"])
        cfg.model.net.dim_in = dim_in + n_met_features

    if use_met and not use_preprocessed and model_key == "lgatr":
        resolved_net = OmegaConf.to_container(cfg.model.net, resolve=True)
        in_s = int(resolved_net.get("in_s_channels", 0))
        cfg.model.net.in_s_channels = in_s + n_met_features
        cfg.model.net.out_s_channels = in_s + n_met_features

    if use_met and not use_preprocessed and model_key == "cnf":
        resolved_net = OmegaConf.to_container(cfg.model.net, resolve=True)
        in_s = int(resolved_net.get("in_s_channels", 0))
        cfg.model.net.in_s_channels = in_s + n_met_features
        # out_s_channels stays as-is so token_dim (derived from it) doesn't shift.

    if use_met and not use_preprocessed and model_key == "lorentznet":
        resolved_net = OmegaConf.to_container(cfg.model.net, resolve=True)
        n_scalar = int(resolved_net["n_scalar"])
        cfg.model.net.n_scalar = n_scalar + n_met_features

    # The Transformer's structured input projection routes input scalars
    # (theta + preprocessed + met) through a separate branch from the
    # particle 4-momentum, and only the latter feeds the per-head vector
    # slots. The wrapper concatenates [scalars | 4-momentum] with the 4-vector
    # last, so we pass the count of input 4-vectors (1 in particles mode, 0
    # in features mode) and the per-head scalar/vector layout. These also
    # determine LLoCa attention's transport, but the structured layer applies
    # regardless of LLoCa being active.
    if model_key == "transformer":
        n_input_vectors = 1 if transformer_input == "particles" else 0
        lloca_block = cfg.model.get("LLoCa", {}) or {}
        n_head_scalars = int(lloca_block.get("LLoCa_num_scalars", 0) or 0)
        n_head_vectors = int(lloca_block.get("LLoCa_num_vectors", 0) or 0)
        if n_input_vectors == 0:
            # Features mode has no input 4-vector to populate vector slots.
            n_head_vectors = 0
        cfg.model.net.n_input_vectors = n_input_vectors
        cfg.model.net.lloca_num_scalars = n_head_scalars
        cfg.model.net.lloca_num_vectors = n_head_vectors

    # Load the right dataset corresponding to model type.
    # MLP and histos models use feature-level inputs while LGATr and
    # Transformer use particles by default.
    # For preprocessing, use the dataset key directly (e.g., 1d, 3d)
    if cfg.exp.key == "preprocessing":
        dataset_key = cfg.dataset.key
    else:
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

    if use_preprocessed and model_key in ("transformer", "lgatr", "lorentznet"):
        run_model_key = f"{run_model_key}_preprocessed"

    cfg.data.run_model_key = run_model_key
    cfg.data.run_dir = (
        f"{cfg.data.run_dir_base}/{cfg.dataset.key}/{cfg.exp.key}/{run_model_key}/{cfg.data.run}"
    )

    OmegaConf.set_struct(cfg, True)

    return cfg
