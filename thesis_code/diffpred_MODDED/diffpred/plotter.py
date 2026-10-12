#!/usr/bin/env python
"""Plot Tox21 val PR-AUC curves from finished Colab / Hydra run dirs.

Each seed is one color. Dashed = best checkpoint, dotted = early stop,
star = test PR-AUC of that checkpoint. Y-axis is zoomed to the data.

Works without results.yaml. Typical Colab layout:

  outputs_jku_chemprop_seed0/wandb/run-YYYYMMDD_HHMMSS-<id>/files/
    config.yaml
    output.log
    wandb-summary.json

Usage (from the diffpred/ folder):
  python plotter.py --glob "outputs_jku_chemprop_seed*" --out plots
  python plotter.py --glob "outputs_jku_chemprop_seed*" --out plots --entity YOUR_WANDB_USER
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

TOX21_TASKS = [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
DEFAULT_PROJECT = "mf-prediction"


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _nested_get(cfg: dict, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
        if isinstance(cur, dict) and set(cur.keys()) <= {"value"}:
            cur = cur.get("value")
    return default if cur is None else cur


def _as_metric_dict(block) -> dict:
    if isinstance(block, list) and block:
        return dict(block[0])
    if isinstance(block, dict):
        return dict(block)
    return {}


def wandb_run_dirs(run_dir: Path) -> list[Path]:
    dirs = []
    for pattern in ("wandb/run-*", "wandb/offline-run-*"):
        dirs.extend(p for p in run_dir.glob(pattern) if p.is_dir())
    latest = run_dir / "wandb" / "latest-run"
    if latest.exists():
        dirs.append(latest.resolve() if latest.is_symlink() else latest)
    return sorted(set(dirs), key=lambda p: p.stat().st_mtime, reverse=True)


def wandb_files_dir(run_dir: Path) -> Path | None:
    for d in wandb_run_dirs(run_dir):
        files = d / "files"
        if files.is_dir():
            return files
    return None


def load_run_config(run_dir: Path) -> dict:
    files = wandb_files_dir(run_dir)
    candidates = [
        run_dir / ".hydra" / "config.yaml",
        run_dir / "config.yaml",
    ]
    if files:
        candidates.insert(0, files / "config.yaml")
    for path in candidates:
        if path.is_file():
            return _read_yaml(path)
    return {}


def load_summary(run_dir: Path) -> dict:
    files = wandb_files_dir(run_dir)
    candidates = []
    if files:
        candidates.append(files / "wandb-summary.json")
    candidates.extend(sorted(run_dir.glob("**/wandb-summary.json")))
    for path in candidates:
        if path.is_file():
            return _read_json(path)
    return {}


def load_results(run_dir: Path) -> dict:
    yaml_path = run_dir / "results.yaml"
    if yaml_path.is_file():
        raw = _read_yaml(yaml_path)
        return {
            "train": _as_metric_dict(raw.get("train")),
            "val": _as_metric_dict(raw.get("val")),
            "test": _as_metric_dict(raw.get("test")),
        }

    summary = load_summary(run_dir)
    if not summary:
        raise FileNotFoundError(
            f"No results.yaml or wandb-summary.json in {run_dir}"
        )
    return {
        "train": {k: v for k, v in summary.items() if str(k).startswith("train/")},
        "val": {k: v for k, v in summary.items() if str(k).startswith("val/")},
        "test": {
            k: v
            for k, v in summary.items()
            if str(k).startswith("test_0/") or str(k).startswith("test/")
        },
    }


def _run_id_from_dirname(name: str) -> str | None:
    match = re.search(r"(?:offline-)?run-\d{8}_\d{6}-([A-Za-z0-9]+)$", name)
    return match.group(1) if match else None


def load_wandb_meta(run_dir: Path, cfg: dict) -> dict:
    meta_path = run_dir / "wandb_run_metadata.yaml"
    if meta_path.is_file():
        meta = _read_yaml(meta_path)
        if meta.get("run_id") and meta.get("project"):
            return meta

    files = wandb_files_dir(run_dir)
    metadata = {}
    if files:
        wb_meta = files / "wandb-metadata.json"
        if wb_meta.is_file():
            metadata = _read_json(wb_meta)

    run_id = metadata.get("id") or metadata.get("run_id")
    if not run_id:
        for d in wandb_run_dirs(run_dir):
            run_id = _run_id_from_dirname(d.name)
            if run_id:
                break

    project = (
        metadata.get("project")
        or _nested_get(cfg, "_wandb", "project")
        or DEFAULT_PROJECT
    )
    entity = metadata.get("entity") or _nested_get(cfg, "_wandb", "entity")
    return {"run_id": run_id, "project": project, "entity": entity}


def _history_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"{path} has no epoch column")
    return (
        df.dropna(subset=["epoch"])
        .sort_values("epoch")
        .groupby("epoch", as_index=False)
        .first()
    )


def _history_from_events_jsonl(path: Path) -> pd.DataFrame | None:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = rec.get("data") if isinstance(rec, dict) and "data" in rec else rec
        if isinstance(payload, dict) and (
            "val/pr_auc" in payload or "epoch" in payload
        ):
            rows.append(payload)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "val/pr_auc" not in df.columns or "epoch" not in df.columns:
        return None
    return (
        df.dropna(subset=["val/pr_auc"])
        .sort_values("epoch")
        .drop_duplicates(subset=["epoch"], keep="first")
        .reset_index(drop=True)
    )


def _history_from_wandb(meta: dict) -> pd.DataFrame:
    import wandb

    entity, project, run_id = meta.get("entity"), meta.get("project"), meta.get("run_id")
    if not (project and run_id):
        raise ValueError("missing W&B project/run_id")
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = wandb.Api().run(path)
    keys = ["epoch", "val/pr_auc", "trainer/global_step"]
    keys += [f"val/pr_auc_task_{i}" for i in range(len(TOX21_TASKS))]
    hist = run.history(keys=keys, pandas=True, samples=20000)
    if hist.empty:
        hist = run.history(pandas=True, samples=20000)
    if "epoch" not in hist.columns:
        raise ValueError(f"W&B run {path} has no epoch column")
    return (
        hist.dropna(subset=["val/pr_auc"])
        .sort_values("epoch")
        .drop_duplicates(subset=["epoch"], keep="first")
        .reset_index(drop=True)
    )


def load_history(run_dir: Path, meta: dict) -> pd.DataFrame:
    csvs = sorted(run_dir.glob("**/metrics.csv"))
    if csvs:
        return _history_from_csv(csvs[0])

    for path in sorted(run_dir.glob("**/*history*.jsonl")) + sorted(
        run_dir.glob("**/wandb-events.jsonl")
    ):
        hist = _history_from_events_jsonl(path)
        if hist is not None and not hist.empty:
            return hist

    if meta.get("run_id"):
        return _history_from_wandb(meta)

    raise FileNotFoundError(
        f"No val history in {run_dir}. "
        "Need metrics.csv, or a synced W&B run (wandb login + online history)."
    )


def infer_seed(run_dir: Path, cfg: dict) -> str:
    seed = _nested_get(cfg, "experiment", "seed")
    if seed is not None:
        return str(seed)
    match = re.search(r"seed[_-]?(\d+)", run_dir.name)
    return match.group(1) if match else run_dir.name


def _seed_sort_key(seed: str):
    return (0, int(seed)) if str(seed).isdigit() else (1, str(seed))


def select_points(hist: pd.DataFrame, cfg: dict, results: dict) -> dict:
    mode = str(_nested_get(cfg, "experiment", "monitor_mode", default="max")).lower()
    monitor = str(_nested_get(cfg, "experiment", "monitor_metric", default="val/pr_auc"))
    if monitor not in hist.columns:
        monitor = "val/pr_auc"
    series = hist[monitor].astype(float)
    best_idx = int(series.idxmax() if mode == "max" else series.idxmin())
    test = results["test"]
    test_score = test.get("test_0/pr_auc", test.get("test/pr_auc"))
    return {
        "best_epoch": float(hist.loc[best_idx, "epoch"]),
        "stop_epoch": float(hist["epoch"].max()),
        "best_val": float(series.loc[best_idx]),
        "test_score": None if test_score is None else float(test_score),
        "test_metrics": test,
        "val_metrics": results["val"],
    }


def seed_colors(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _zoom_ylim(ax, values: list[float], pad_frac: float = 0.12, min_pad: float = 0.02) -> None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        ax.set_ylim(0.0, 1.0)
        return
    lo, hi = min(vals), max(vals)
    pad = max(min_pad, pad_frac * max(hi - lo, 1e-3))
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))


def _draw_seed(ax, run: dict, color, *, label=None, star_size=150):
    hist, pts = run["hist"], run["points"]
    ax.plot(
        hist["epoch"],
        hist["val/pr_auc"],
        color=color,
        lw=1.8,
        marker="o",
        ms=3,
        label=label,
    )
    ax.axvline(pts["best_epoch"], color=color, ls="--", lw=1.0, alpha=0.75)
    if pts["stop_epoch"] != pts["best_epoch"]:
        ax.axvline(pts["stop_epoch"], color=color, ls=":", lw=1.0, alpha=0.55)
    if pts["test_score"] is not None:
        ax.scatter(
            [pts["best_epoch"]],
            [pts["test_score"]],
            color=color,
            marker="*",
            s=star_size,
            zorder=5,
            edgecolors="k",
            linewidths=0.35,
        )


def plot_macro(runs: list[dict], out_dir: Path) -> None:
    colors = seed_colors(len(runs))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y_values: list[float] = []
    for run, color in zip(runs, colors):
        y = run["hist"]["val/pr_auc"].astype(float)
        y_values.extend(y.dropna().tolist())
        test = run["points"]["test_score"]
        if test is not None:
            y_values.append(float(test))
        label = f"seed {run['seed']}"
        if test is not None:
            label += f"  test={test:.3f}"
        _draw_seed(ax, run, color, label=label)

    tests = [r["points"]["test_score"] for r in runs if r["points"]["test_score"] is not None]
    if len(tests) > 1:
        mean, std = float(np.mean(tests)), float(np.std(tests, ddof=1))
        ax.axhline(
            mean,
            color="0.2",
            ls="-.",
            lw=1.1,
            label=f"mean test {mean:.3f} ± {std:.3f}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Tox21 — val PR-AUC by seed")
    _zoom_ylim(ax, y_values)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "val_curve_all_seeds.png", dpi=160)
    plt.close(fig)


def plot_tasks(runs: list[dict], out_dir: Path) -> None:
    task_cols = [f"val/pr_auc_task_{i}" for i in range(len(TOX21_TASKS))]
    if not any(col in run["hist"].columns for run in runs for col in task_cols):
        return

    colors = seed_colors(len(runs))
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.6), sharex=True, sharey=False)
    for i, ax in enumerate(axes.ravel()):
        col = f"val/pr_auc_task_{i}"
        vals: list[float] = []
        for run, color in zip(runs, colors):
            if col not in run["hist"].columns:
                continue
            series = run["hist"][col].dropna().astype(float)
            vals.extend(series.tolist())
            ax.plot(run["hist"]["epoch"], run["hist"][col], color=color, lw=1.3)
            ax.axvline(run["points"]["best_epoch"], color=color, ls="--", lw=0.8, alpha=0.6)
            test_val = run["points"]["test_metrics"].get(f"test_0/pr_auc_task_{i}")
            if test_val is not None:
                vals.append(float(test_val))
                ax.scatter(
                    [run["points"]["best_epoch"]],
                    [float(test_val)],
                    color=color,
                    marker="*",
                    s=70,
                    zorder=5,
                    edgecolors="k",
                    linewidths=0.25,
                )
        ax.set_title(TOX21_TASKS[i], fontsize=10)
        _zoom_ylim(ax, vals)

    handles = [
        plt.Line2D([0], [0], color=color, lw=1.8, label=f"seed {run['seed']}")
        for run, color in zip(runs, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(runs),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.supxlabel("Epoch")
    fig.supylabel("PR-AUC")
    fig.suptitle("Tox21 — per-task val PR-AUC (color = seed, star = test)", y=1.06)
    fig.tight_layout()
    fig.savefig(out_dir / "val_curve_tasks_all_seeds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot val curves + early stop + test")
    p.add_argument("runs", nargs="*", type=Path, help="Run directories")
    p.add_argument("--glob", dest="run_glob", default=None, help='e.g. "outputs_jku_chemprop_seed*"')
    p.add_argument("--out", type=Path, default=Path("plots"))
    p.add_argument("--entity", default=None, help="W&B entity if metadata is missing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = list(args.runs)
    if args.run_glob:
        run_dirs.extend(Path(p) for p in sorted(glob.glob(args.run_glob)))
    run_dirs = [p.resolve() for p in dict.fromkeys(run_dirs) if p.is_dir()]
    if not run_dirs:
        raise SystemExit("Pass run dirs or --glob 'outputs_jku_chemprop_seed*'")

    args.out.mkdir(parents=True, exist_ok=True)
    loaded = []
    for run_dir in run_dirs:
        cfg = load_run_config(run_dir)
        results = load_results(run_dir)
        meta = load_wandb_meta(run_dir, cfg)
        if args.entity:
            meta["entity"] = args.entity
        hist = load_history(run_dir, meta)
        points = select_points(hist, cfg, results)
        seed = infer_seed(run_dir, cfg)
        hist.to_csv(args.out / f"history_seed{seed}.csv", index=False)
        print(
            f"seed {seed}: best_epoch={points['best_epoch']:.0f}  "
            f"stop={points['stop_epoch']:.0f}  "
            f"val={points['best_val']:.4f}  "
            f"test={points['test_score']}"
        )
        loaded.append({"seed": seed, "hist": hist, "points": points})

    loaded.sort(key=lambda r: _seed_sort_key(r["seed"]))
    plot_macro(loaded, args.out)
    plot_tasks(loaded, args.out)

    tests = [r["points"]["test_score"] for r in loaded if r["points"]["test_score"] is not None]
    if len(tests) > 1:
        print(
            f"test PR-AUC: {np.mean(tests):.3f} ± {np.std(tests, ddof=1):.3f}  "
            f"(n={len(tests)})"
        )
    print(f"wrote plots to {args.out.resolve()}")


if __name__ == "__main__":
    main()