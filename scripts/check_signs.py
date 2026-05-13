"""
Sign-convention sanity check for ratio-regression models on the test set.

Loads a trained ratio-regression model (via the same Hydra plumbing used
during training/eval), runs the ratio test loader, and reports four
correlations that should all be positive if no sign was flipped:

    (1) corr(log_r_true,  <delta_theta, score_true>)   -- data sanity
    (2) corr(log_r_pred,  <delta_theta, score_pred>)   -- model output sanity
    (3) corr(log_r_true,  log_r_pred)                  -- ratio matches
    (4) corr(score_true,  score_pred)                  -- score matches (per-component, then averaged)

`delta_theta = theta0 - theta1`, both read from the raw .npy files. For 1d
all four reduce to scalar correlations; for higher-dim theta we use the
inner product (delta_theta . score).

Example:
    python scripts/check_signs.py exp=ratio model=transformer dataset=1d data.run=3
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Make project root importable when running as `python scripts/check_signs.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers.derive_config import derive_config  # noqa: E402


def _resolvers():
    if not OmegaConf.has_resolver("sum"):
        OmegaConf.register_new_resolver("sum", lambda *vs: sum(vs))
    if not OmegaConf.has_resolver("prod"):
        import math
        OmegaConf.register_new_resolver(
            "prod", lambda *vs: math.prod(int(v) for v in vs)
        )
    if not OmegaConf.has_resolver("env"):
        import os
        OmegaConf.register_new_resolver(
            "env",
            lambda key: {"prefix": Path(sys.executable).parent, "cwd": os.getcwd()}.get(key),
        )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(overrides: list[str]) -> None:
    _resolvers()

    conf_dir = str((ROOT / "conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=overrides)
    cfg = derive_config(cfg)

    # We only want eval; suppress accidental training.
    OmegaConf.set_struct(cfg, False)
    cfg.modes.train = False
    cfg.modes.eval = False
    cfg.modes.plot = False
    cfg.modes.recycle = True
    OmegaConf.set_struct(cfg, True)

    # Build experiment, run the prefix of run() that loads checkpoint, model,
    # normalizer, datasets and loaders -- but stop short of training/eval.
    exp = instantiate(cfg.exp)(cfg=cfg)
    exp.init()
    exp._init()

    if exp.checkpoints.state_dict is None:
        raise SystemExit(
            f"No checkpoint found at {Path(cfg.data.run_dir) / cfg.data.ckpts}"
        )

    # Read theta1 (reference) directly; the dataset loader keeps only theta0.
    src = Path(cfg.dataset.path)
    theta1_test = np.load(src / "theta1_test_ratio.npy")  # (N, theta_dim)

    model = exp.model
    model.eval()

    log_r_true_all, log_r_pred_all = [], []
    score_true_all, score_pred_all = [], []
    theta0_all = []

    n_seen = 0
    with torch.enable_grad():
        for batch in exp.test_loader:
            batch.to_(**exp.device_kwds)
            # Force theta to require grad so we can take d log_r / d theta.
            batch.theta.requires_grad_(True)

            # Mirror ExperimentRatiosParticles._preds() forward pass.
            embedding_kwargs = {
                "theta": batch.theta,
                "ptr": batch.ptr,
                "mode": cfg.model.get("mode", "channels"),
                "theta_dim": batch.theta.shape[-1],
            }
            if exp._use_preprocessed:
                embedding_kwargs["preprocessed"] = batch.preprocessed
            if exp._use_met:
                embedding_kwargs["met"] = batch.met

            model_kwds = {
                "particles": batch.particles,
                "ptr": batch.ptr,
                "force_math": True,  # needed for double-backward through attention
                "embedding_kwargs": embedding_kwargs,
            }
            if cfg.model.key == "lgatr":
                model_kwds["scalars"] = batch.preprocessed if exp._use_preprocessed else None
                model_kwds["met"] = batch.met if exp._use_met else None

            log_r_pred = model(**model_kwds)  # (B, 1)
            (score_pred,) = torch.autograd.grad(
                log_r_pred,
                batch.theta,
                grad_outputs=torch.ones_like(log_r_pred),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )

            log_r_pred_all.append(log_r_pred.detach().reshape(-1).cpu().numpy())
            score_pred_all.append(score_pred.detach().cpu().numpy())

            # Targets: log r_true comes from r_xz (>0 by construction).
            log_r_true_all.append(
                torch.log(batch.ratio.clamp_min(1e-30)).detach().reshape(-1).cpu().numpy()
            )
            score_true_all.append(batch.score.detach().cpu().numpy())
            theta0_all.append(batch.theta.detach().cpu().numpy())

            n_seen += batch.theta.shape[0]

    log_r_true = np.concatenate(log_r_true_all)
    log_r_pred = np.concatenate(log_r_pred_all)
    score_true = np.concatenate(score_true_all, axis=0)  # (N, theta_dim)
    score_pred = np.concatenate(score_pred_all, axis=0)
    theta0     = np.concatenate(theta0_all, axis=0)
    theta1     = theta1_test[: theta0.shape[0]]

    delta_theta = theta0 - theta1                    # (N, theta_dim)
    proj_true = np.sum(delta_theta * score_true, axis=1)
    proj_pred = np.sum(delta_theta * score_pred, axis=1)

    c_true_proj  = _pearson(log_r_true, proj_true)
    c_pred_proj  = _pearson(log_r_pred, proj_pred)
    c_logr       = _pearson(log_r_true, log_r_pred)
    c_score_per  = [_pearson(score_true[:, i], score_pred[:, i])
                    for i in range(score_true.shape[1])]

    print()
    print(f"run_dir : {cfg.data.run_dir}")
    print(f"events  : {n_seen}")
    print(f"theta_dim: {score_true.shape[1]}")
    print()
    print("All four should be POSITIVE; a negative one points at a flipped sign.")
    print(f"  corr(log_r_true, <delta_theta, score_true>) = {c_true_proj:+.4f}")
    print(f"  corr(log_r_pred, <delta_theta, score_pred>) = {c_pred_proj:+.4f}")
    print(f"  corr(log_r_true, log_r_pred)                = {c_logr:+.4f}")
    for i, c in enumerate(c_score_per):
        print(f"  corr(score_true[{i}], score_pred[{i}])           = {c:+.4f}")

    flipped = []
    for name, val in [
        ("log_r_true vs <dtheta,score_true>", c_true_proj),
        ("log_r_pred vs <dtheta,score_pred>", c_pred_proj),
        ("log_r_true vs log_r_pred",          c_logr),
    ] + [(f"score component {i}", c) for i, c in enumerate(c_score_per)]:
        if not np.isnan(val) and val < 0:
            flipped.append(name)

    print()
    if flipped:
        print("NEGATIVE correlations (likely flipped sign):")
        for f in flipped:
            print(f"  - {f}")
    else:
        print("All correlations are non-negative.")


if __name__ == "__main__":
    main(sys.argv[1:])
