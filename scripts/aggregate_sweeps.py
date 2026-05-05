"""Aggregate per-model wandb HPO sweep results into one comparison table.

For each sweep under a project (or each id listed in a sweep_ids.json saved
by ExperimentSweep), pulls the best `--top-n` runs by `loss/val_best` and
prints a comparison table with the searched hyperparameters.

Examples:
    # Use the JSON map saved by a sweep run
    python scripts/aggregate_sweeps.py \\
        --ids run_dirs/1d/sweep/transformer/<run>/sweep_ids.json

    # Discover sweeps by name pattern across the whole project
    python scripts/aggregate_sweeps.py \\
        --project tzq-sbi-ml --pattern hpo-

    # Save as CSV for downstream analysis
    python scripts/aggregate_sweeps.py --project tzq-sbi-ml --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import wandb


# Hparam keys we surface in the output table (in display order).
HPARAM_KEYS = (
    "train.lr",
    "train.dropout",
    "train.batch_size",
    "train.weight_decay",
    "train.clip_grad_norm",
    "train.lr_warmup",
    "train.lr_min",
    "train.lr_restart_period",
    "train.lr_restart_mult",
    "loss.log_r_clip",
)
METRIC_KEY = "loss/val_best"


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _flat_get(cfg: dict, key: str) -> Any:
    """Look up a dotted-path key in a wandb run config (supports flat or nested)."""
    if key in cfg:
        return cfg[key]
    cur: Any = cfg
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _resolve_sweeps(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Return [(label, project_path, sweep_id), ...]."""
    if args.ids:
        path = Path(args.ids)
        payload = json.loads(path.read_text())
        project = payload.get("project") or args.project
        entity = payload.get("entity") or args.entity
        if not project:
            sys.exit("ids file has no 'project' and --project not given")
        scope = f"{payload.get('dataset', '?')}-{payload.get('exp', '?')}"
        proj_path = f"{entity}/{project}" if entity else project
        return [
            (f"{scope}-{preset}", proj_path, sid)
            for preset, sid in payload["sweep_ids"].items()
        ]

    if not args.project:
        sys.exit("must pass either --ids or --project")
    proj_path = f"{args.entity}/{args.project}" if args.entity else args.project
    api = wandb.Api()
    sweeps = list(api.project(args.project, entity=args.entity).sweeps())
    matched = [s for s in sweeps if (not args.pattern) or args.pattern in s.name]
    if not matched:
        sys.exit(f"No sweeps in {proj_path} matching pattern={args.pattern!r}")
    return [(s.name, proj_path, s.id) for s in matched]


def _best_runs(
    api: wandb.Api,
    proj_path: str,
    sweep_id: str,
    top_n: int,
    metric: str,
) -> list[wandb.apis.public.Run]:
    sweep = api.sweep(f"{proj_path}/{sweep_id}")
    runs = [r for r in sweep.runs if r.state == "finished"]
    runs.sort(key=lambda r: r.summary.get(metric, float("inf")))
    return runs[:top_n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", help="Path to sweep_ids.json saved by ExperimentSweep")
    p.add_argument("--project", help="Wandb project name (used if --ids not given)")
    p.add_argument("--entity", default=None)
    p.add_argument("--pattern", default="hpo-", help="Substring filter for sweep discovery")
    p.add_argument("--top-n", type=int, default=1, help="Best N runs per sweep (default 1)")
    p.add_argument("--metric", default=METRIC_KEY)
    p.add_argument("--csv", default=None, help="Optional CSV output path")
    args = p.parse_args()

    triples = _resolve_sweeps(args)
    api = wandb.Api()

    rows: list[dict[str, Any]] = []
    for label, proj_path, sweep_id in triples:
        try:
            best = _best_runs(api, proj_path, sweep_id, args.top_n, args.metric)
        except Exception as exc:
            print(f"[warn] could not fetch sweep {label} ({sweep_id}): {exc}", file=sys.stderr)
            continue
        if not best:
            print(f"[warn] sweep {label} ({sweep_id}) has no finished runs", file=sys.stderr)
            continue
        for rank, run in enumerate(best, start=1):
            row: dict[str, Any] = {
                "sweep": label,
                "rank": rank,
                "run_id": run.id,
                "metric": run.summary.get(args.metric),
            }
            for k in HPARAM_KEYS:
                row[k] = _flat_get(run.config, k)
            rows.append(row)

    if not rows:
        sys.exit("No rows aggregated.")

    rows.sort(key=lambda r: (r["sweep"], r["rank"]))

    columns = ["sweep", "rank", "metric", *HPARAM_KEYS, "run_id"]
    widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    line = " | ".join(c.ljust(widths[c]) for c in columns)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in columns))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
