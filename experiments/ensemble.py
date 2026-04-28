"""Enseble class to aggregate results from different experiments and runs"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from .base.schemas import Chekcpoints
from .logger import LOGGER as _LOGGER
from .plotting import (
    AttentionExtractor,
    plot_attention_maps,
    plot_attention_summary,
    plot_intervals,
    plot_llr,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

LOGGER = _LOGGER.getChild(__name__)

# Translate model strings into labels appropriate to display on a plot
# Asume empty string to be histograms (no model)
MODEL2LABEL = {
    "mlp": "MLP",
    "lgatr": "LGATr",
    "transformer": "Transformer",
    "lorentznet": "LorentzNet",
    "": "2D Histogram",
}

# Translate experiment keys into human-readable method labels
EXP2LABEL = {
    "local": "Score Regression (SALLY)",
    "ratio": "Likelihood Ratio Regression (ROLR)",
    "histo": "Histogram",
}

# Filesystem-friendly directory names for each experiment key
EXP2DIRNAME = {
    "local": "score_regression",
    "ratio": "ratio_regression",
    "histo": "histogram",
}

# For ploting LLR contours
COLORS = ["#20908C", "#D4BB36", "#A43FA0", "#4C72B0"]

# Number of dummy particles used when extracting attention maps
_N_DUMMY_PARTICLES = 5


def _build_model_and_dummy(
    model_key: str, state_dict: Dict
) -> tuple:
    """
    Reconstruct the inner model (Transformer or L-GATr) from a checkpoint
    state_dict and return it together with a matching dummy input tensor.

    The wrapper stores the inner network under the ``net.`` prefix, so all
    keys are stripped of that prefix before loading.

    :param model_key: ``"transformer"`` or ``"lgatr"``
    :param state_dict: Full wrapper state dict (keys prefixed with ``net.``)
    :return: ``(model, dummy_input)``
    """
    # Strip "net." prefix to get the inner-model state dict
    inner_sd = {
        k.removeprefix("net."): v
        for k, v in state_dict.items()
        if k.startswith("net.")
    }

    if model_key == "transformer":
        return _build_transformer(inner_sd)
    elif model_key == "lgatr":
        return _build_lgatr(inner_sd)
    else:
        raise ValueError(f"Unsupported model key for attention: {model_key}")


def _build_transformer(inner_sd: Dict) -> tuple:
    """Reconstruct a :class:`Transformer` from its state dict."""
    from models.transformer import Transformer, derive_emb_hidden
    from models.configs import SAConfig

    # Infer dimensions from weight matrices
    dim_in = inner_sd["linear_in.weight"].shape[1]
    emb_hidden = inner_sd["linear_in.weight"].shape[0]
    dim_out = inner_sd["linear_out.weight"].shape[0]

    # Count transformer encoder blocks
    block_ids = {
        int(k.split(".")[1])
        for k in inner_sd
        if k.startswith("te_blocks.")
    }
    num_blocks = len(block_ids)

    # Read num_heads from the config yaml (default: 8)
    num_heads = 8
    dropout_p = 0.0
    try:
        from helpers.derive_config import load_conf_from
        mcfg = load_conf_from(Path("conf/model/transformer"))
        num_heads = mcfg.get("net", {}).get("attention", {}).get("num_heads", 8)
        dropout_p = mcfg.get("train", {}).get("dropout", 0.0)
    except Exception:
        pass

    # If any MLP linear lives at index 3 (Linear-ReLU-Dropout pattern)
    # dropout was used during training; match it so Sequential indices align
    if any("mlp.net.3.weight" in k for k in inner_sd):
        dropout_p = max(dropout_p, 0.1)

    # Compute emb_factor that reproduces the same emb_hidden
    emb_factor = 1
    while derive_emb_hidden(dim_in, emb_factor, num_heads) < emb_hidden:
        emb_factor += 1

    # SAConfig supports attribute access (attention.num_heads),
    # which the Transformer constructor requires before casting
    model = Transformer(
        dim_in=dim_in,
        emb_factor=emb_factor,
        dim_out=dim_out,
        num_blocks=num_blocks,
        attention=SAConfig(num_heads=num_heads),
        mlp={"k_factor": 2, "activation": "relu", "dim_in": emb_hidden, "dim_out": emb_hidden},
        dropout_p=dropout_p,
    )
    model.load_state_dict(inner_sd)

    # 3-D dummy so MultiHA sees a real sequence dimension
    dummy = torch.randn(1, _N_DUMMY_PARTICLES, dim_in)
    return model, dummy


def _build_lgatr(inner_sd: Dict) -> tuple:
    """Reconstruct an :class:`lgatr.LGATr` net from its state dict."""
    from lgatr import LGATr

    # LGATr stores per-block parameters under "blocks.{i}.*"
    block_ids = {
        int(k.split(".")[1])
        for k in inner_sd
        if k.startswith("blocks.")
    }
    num_blocks = len(block_ids)

    # Read channel sizes from config yaml (defaults from lgatr.yaml)
    hidden_mv = 8
    hidden_s = 16
    in_s = out_s = 0
    num_heads = 2
    try:
        from helpers.derive_config import load_conf_from
        mcfg = load_conf_from(Path("conf/model/lgatr"))
        net = mcfg.get("net", {})
        hidden_mv = net.get("hidden_mv_channels", hidden_mv)
        hidden_s = net.get("hidden_s_channels", hidden_s)
        in_s = net.get("in_s_channels", in_s)
        out_s = net.get("out_s_channels", out_s)
        num_heads = net.get("attention", {}).get("num_heads", num_heads)
    except Exception:
        pass

    # Infer in/out mv channels from the GA EquiLinear weight tensors.
    # These have 3-D shape (out_mv_channels, in_mv_channels, basis_size)
    # and are stored under exactly "linear_in.weight" / "linear_out.weight"
    # (NOT the sub-module keys like "linear_in.mvs2s.weight").
    in_mv = inner_sd["linear_in.weight"].shape[1]   # (hidden_mv, in_mv, 10)
    out_mv = inner_sd["linear_out.weight"].shape[0]  # (out_mv, hidden_mv, 10)

    # Infer in_s from the scalar-to-scalar projection (may be 0)
    if "linear_in.s2s.weight" in inner_sd:
        in_s = inner_sd["linear_in.s2s.weight"].shape[1]
    if "linear_out.s2s.weight" in inner_sd:
        out_s = inner_sd["linear_out.s2s.weight"].shape[0]

    model = LGATr(
        in_mv_channels=in_mv,
        out_mv_channels=out_mv,
        hidden_mv_channels=hidden_mv,
        in_s_channels=in_s,
        out_s_channels=out_s,
        hidden_s_channels=hidden_s,
        num_blocks=num_blocks,
        attention={"num_heads": num_heads},
        mlp={},
        dropout_prob=0.0,
    )
    model.load_state_dict(inner_sd)

    # Multivector dummy: (batch, num_particles, in_mv_channels, 16)
    dummy = torch.randn(1, _N_DUMMY_PARTICLES, in_mv, 16)
    return model, dummy


class Ensemble:
    """
    Class to merge results from different expeirments and runs
    """

    def __init__(
        self, *, cfg: DictConfig, checkpoints_from: Dict, key: str = "ensemble"
    ) -> None:
        self.cfg = cfg
        self.key = key
        self.checkpoints_from = checkpoints_from

    def _resolve_run_model_dir(self, model_key: str) -> str:
        """Resolve model directory name used under run_dirs for checkpoints."""
        if model_key != "transformer":
            return model_key

        lloca_active = False
        if self.cfg.model.get("key", None) == "transformer":
            lloca_cfg = self.cfg.model.get("LLoCa", {})
            lloca_active = lloca_cfg.get("active", False)

        return "transformer_lloca" if lloca_active else "transformer"

    def run(self) -> None:
        """
        Iterate over: experiment, model and run to create merged plots

        ::note:: In the future, grid, parameter names, ranges and resolutions
            will be defined in a file and loaded from there to avoid confusion

        """
        # Containers for the log-likelihood ratio arrays of predictions
        llr_all, std_all = [], []

        # Translate model string into a label appropriate for plotting
        labels_all = []

        # Derive a human-readable method label from the experiment keys
        exp_keys = set(self.checkpoints_from["exp"])
        # Use the ML method (non-histo) if available, otherwise "Histogram"
        ml_keys = exp_keys - {"histo"}
        method = ", ".join(EXP2LABEL.get(k, k) for k in sorted(ml_keys)) if ml_keys else EXP2LABEL["histo"]

        # Build output directory: images/<dataset>/<exp_key>/<method_dirname>/
        exp_dir_key = sorted(ml_keys)[0] if ml_keys else "histo"
        method_dirname = EXP2DIRNAME.get(exp_dir_key, exp_dir_key)
        out_dir = Path("images", self.cfg.dataset.key, exp_dir_key, method_dirname)
        out_dir.mkdir(parents=True, exist_ok=True)

        # These should be loaded from a file ...
        grid = names = ranges = resolutions = None

        # Iterate over all combinations of exp, model and run possible
        for exp, model, runs in zip(*self.checkpoints_from.values()):

            # Container for the LLR of one combination of model and exp
            llr_this = []

            # Iterate over all runs
            for run in runs:
                run_model_dir = self._resolve_run_model_dir(model)
                ckpts_path = Path(
                    self.cfg.data.run_dir_base,
                    self.cfg.dataset.key,
                    exp,
                    run_model_dir,
                    str(run),
                    self.cfg.data.ckpts,
                )

                # Load checkpoints
                checkpoints = Chekcpoints(
                    **torch.load(ckpts_path, map_location="cpu", weights_only=False)
                )

                llr_this.append(checkpoints.limits.llr)
                print(
                    f"Exp: {exp} model {model} resolutions {checkpoints.limits.resolutions}"
                )

                # Since we are not storing these in a file, we have to check for
                # consistency. FIXME
                new_names = checkpoints.limits.param_names
                new_ranges = checkpoints.limits.ranges
                new_resolutions = checkpoints.limits.resolutions

                # Treat empty lists as unset (some checkpoints lack param_names)
                if names and new_names:
                    assert names == new_names, f"param_names mismatch: {names} vs {new_names}"
                if ranges is not None and new_ranges is not None:
                    assert np.all(ranges == new_ranges)
                if resolutions is not None and new_resolutions is not None:
                    assert np.all(resolutions == new_resolutions)

                # Keep the most informative value
                grid = checkpoints.limits.grid
                names = new_names if new_names else names
                ranges = new_ranges if new_ranges is not None else ranges
                resolutions = new_resolutions if new_resolutions is not None else resolutions

            # Average over all runs
            mean_llr = np.asarray(llr_this).mean(axis=0)

            # Rescale the averaged LLR
            mean_llr = mean_llr - mean_llr.max()

            # Append results to global containers
            llr_all.append(mean_llr)
            std_all.append(np.asarray(llr_this).std(axis=0))

            # Keep track of which model produced which results
            labels_all.append(MODEL2LABEL[model])

        # Fallback: if no checkpoint stored param_names, generate generic ones
        if names is None and grid is not None:
            names = [f"param_{i}" for i in range(grid.shape[1])]

        plot_llr(
            llr_list=llr_all,
            std_list=std_all,
            param_names=names,
            grid=grid,
            ranges=ranges,
            resolutions=resolutions,
            labels=labels_all,
            to=str(out_dir / f"{method_dirname}.png"),
            conf_levels=(0.68,),
            colors=COLORS,
            method=method,
        )
        plot_intervals(llr_all, grid, labels_all, to=str(out_dir / f"{method_dirname}_limits.png"), colors=COLORS, method=method)

        # Plot attention maps for attention-based models
        self._plot_attention(out_dir, method_dirname)

    def _plot_attention(self, out_dir: Path, method_dirname: str) -> None:
        """
        For each attention-capable model (Transformer, L-GATr) in the ensemble,
        reconstruct the model from the first run's checkpoint, run a dummy
        forward pass, and plot the extracted attention maps.
        """
        for exp, model_key, runs in zip(*self.checkpoints_from.values()):
            if model_key not in ("transformer", "lgatr"):
                continue

            run_model_dir = self._resolve_run_model_dir(model_key)

            ckpts_path = Path(
                self.cfg.data.run_dir_base,
                self.cfg.dataset.key,
                exp,
                run_model_dir,
                str(runs[0]),
                self.cfg.data.ckpts,
            )

            if not ckpts_path.exists():
                LOGGER.warning(f"Checkpoint not found for {model_key}: {ckpts_path}")
                continue

            checkpoints = Chekcpoints(
                **torch.load(ckpts_path, map_location="cpu", weights_only=False)
            )

            if checkpoints.state_dict is None:
                LOGGER.warning(f"No state_dict in checkpoint for {model_key}")
                continue

            try:
                model, dummy = _build_model_and_dummy(
                    model_key, checkpoints.state_dict
                )
                model.eval()

                extractor = AttentionExtractor(model)
                with torch.no_grad():
                    model(dummy) if model_key == "transformer" else model(
                        multivectors=dummy
                    )
                attn_maps = extractor.get()
                extractor.remove()

                if not attn_maps:
                    LOGGER.warning(
                        f"No attention maps captured for {model_key}"
                    )
                    continue

                label = MODEL2LABEL[model_key]
                plot_attention_maps(
                    attn_maps,
                    model_name=label,
                    to=str(out_dir / f"{method_dirname}.png"),
                )
                plot_attention_summary(
                    attn_maps,
                    model_name=label,
                    to=str(out_dir / f"{method_dirname}.png"),
                )
                LOGGER.info(f"Attention maps plotted for {label}")

            except Exception as e:
                LOGGER.warning(
                    f"Could not extract attention maps for {model_key}: {e}"
                )

    def __call__(self):
        self.run()
