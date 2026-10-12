#!/usr/bin/env python
"""
See TUNING_CHEMPROP.md for details
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
TOX21_DIR = PACKAGE_DIR / "exp_configs" / "tox21"
IMPLEMENTATION = "DiffPred ChempropMPNNet with its original readout and Adam/ReduceLROnPlateau"
METRICS = {"pr_auc": "max", "roc_auc": "max", "loss": "min"}
HPARAMS = {
    "chemprop_depth", "feature_hiddens", "output_mlp_type",
    "num_hiddens", "dropout", "activation", "chemprop_message_passing",
    "chemprop_aggregation",
}
TRAINING = {
    "batch_size", "lr", "weight_decay", "use_scheduler",
    "lr_schedule_factor", "lr_schedule_patience",
}
SHARED_ENCODER = {
    "net_type", "feature_hiddens", "chemprop_depth", "chemprop_message_passing",
    "chemprop_aggregation", "activation", "dropout",
}


def training_defaults(base):
    """Make only existing NetTrainer/reloader defaults explicit."""
    base = copy.deepcopy(base)
    for key, value in {"weight_decay": 0.0, "use_scheduler": True,
                       "lr_schedule_factor": 0.5, "lr_schedule_patience": 20}.items():
        base["model"]["training"].setdefault(key, value)
    return base


def read_yaml(path):
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_yaml(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(yaml.safe_dump(data, sort_keys=False))
    temp.replace(path)


def positive(value, name, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


def validate_config(cfg):
    h, t, e, d = cfg["model"]["hparams"], cfg["model"]["training"], cfg["experiment"], cfg["data"]
    if (d["name"] != "jku_tox21_splits" or not d.get("classification")
            or d["diffusion_params"]["diffusion_type"] is not None
            or h["net_type"] != "chemprop" or h.get("arch_type") is not None
            or cfg["model"]["featurizer"] != "graph" or e.get("debug")
            or any(cfg["model"].get(k) for k in ("use_mve", "use_mcdropout", "use_deepensemble"))):
        raise ValueError("Use a non-debug JKU Chemprop classification baseline")
    if set(h) & {"chemprop_readout_type", "chemprop_readout_hidden_dim"}:
        raise ValueError("Use DiffPred's chemprop.yaml; custom JKU readout options were removed")
    if set(t) & {"scheduler_name", "max_lr", "final_lr", "warmup_epochs", "optimizer_name",
                 "adam_beta1", "adam_beta2", "adam_eps", "min_lr"}:
        raise ValueError("Use the existing DiffPred Adam/plateau training settings")
    widths = h["feature_hiddens"]
    if not isinstance(widths, list) or len(widths) != 1:
        raise ValueError("feature_hiddens must contain exactly one message-passing width")
    positive(widths[0], "feature_hiddens[0]", True)
    for key in ("chemprop_depth", "num_hiddens"):
        positive(h[key], key, True)
    if widths[0] // 2 ** (h["num_hiddens"] - 1) < 1:
        raise ValueError("Readout depth would create a zero-width layer")
    for key in ("batch_size", "max_epochs"):
        positive(t[key], key, True)
    positive(t["lr"], "lr")
    if not 0 <= h["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not math.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 < t["lr_schedule_factor"] < 1:
        raise ValueError("lr_schedule_factor must be in (0, 1)")
    positive(t["lr_schedule_patience"], "lr_schedule_patience", True)
    if not isinstance(t["use_scheduler"], bool):
        raise ValueError("use_scheduler must be true or false")
    for value, allowed, key in (
        (h["activation"], {"relu", "silu", "tanh"}, "activation"),
        (h["chemprop_aggregation"], {"mean", "sum", "norm"}, "aggregation"),
        (h["chemprop_message_passing"], {"bond", "atom"}, "message passing"),
        (h["output_mlp_type"], {"linear", "residual"}, "output_mlp_type"),
    ):
        if value not in allowed:
            raise ValueError(f"Unsupported {key}: {value}")


def build_candidates(base, candidates):
    base = training_defaults(base)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("The candidates YAML must contain a nonempty candidates list")
    names, signatures, result = set(), set(), []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be a mapping with a name")
        name = candidate.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"Candidate names must be unique and path-safe: {name!r}")
        names.add(name)
        if set(candidate) - {"name", "hparams", "training"}:
            raise ValueError(f"Unknown sections in candidate {name}")
        cfg = copy.deepcopy(base)
        for section, allowed in (("hparams", HPARAMS), ("training", TRAINING)):
            overrides = candidate.get(section, {})
            if not isinstance(overrides, dict) or set(overrides) - allowed:
                raise ValueError(f"Unsupported {section} overrides in {name}: {overrides}")
            cfg["model"][section].update(overrides)
        validate_config(cfg)
        signature = json.dumps(cfg, sort_keys=True)
        if signature in signatures:
            raise ValueError(f"Duplicate effective configuration: {name}")
        signatures.add(signature)
        result.append((name, cfg))
    return result


def final_configs(best, template):
    """Transfer the encoder and optimization"""
    baseline, diffusion = copy.deepcopy(best), copy.deepcopy(template)
    diff_params = copy.deepcopy(diffusion["data"]["diffusion_params"])
    diffusion["data"] = copy.deepcopy(baseline["data"])
    diffusion["data"].update(diffusion_params=diff_params, standardize_targets=True,
                             symmetric_binary_targets=True)
    for key in SHARED_ENCODER:
        diffusion["model"]["hparams"][key] = copy.deepcopy(baseline["model"]["hparams"][key])
    # num_hiddens and output_mlp_type describe the baseline head, not the conditioned
    # diffusion head. The latter stays exactly as specified in the template.
    for key in TRAINING | {"max_epochs"}:
        diffusion["model"]["training"][key] = baseline["model"]["training"][key]
    diffusion_ema = diffusion["experiment"].get("use_ema_on_test", True)
    diffusion["experiment"] = copy.deepcopy(baseline["experiment"])
    diffusion["experiment"]["use_ema_on_test"] = diffusion_ema
    baseline["data"].update(standardize_targets=False, symmetric_binary_targets=False)
    for cfg, suffix in ((baseline, ""), (diffusion, "_diff")):
        cfg["data"]["load_test"] = True
        cfg["experiment"].update(
            tune=False, debug=False, wandb_mode="disabled",
            exp_name="jku_chemprop" + suffix + "_seed${experiment.seed}",
            output_dir="outputs_jku_chemprop" + suffix + "_seed${experiment.seed}",
        )
    return baseline, diffusion


def output_root(path):
    if path is None:
        path = Path("outputs_tune/tox21") / datetime.now().strftime("diffpred_manual_%Y%m%d_%H%M%S_%f")
    path = path.expanduser()
    if not path.is_absolute():
        root = os.environ.get("DIFFPRED_OUTPUT_ROOT")
        if root is None and Path("/content/drive/MyDrive").exists():
            root = "/content/drive/MyDrive/AI_thesis/diffpred_runs"
        path = Path(root or Path.cwd()) / path
    return path.resolve()


def data_fingerprint(directory):
    """Read only train/valid"""
    result = {}
    for split, names in (("train", ["train.csv"]), ("valid", ["valid.csv", "val.csv"])):
        path = next((directory / n for n in names if (directory / n).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {split} CSV in {directory}; pass --data-dir")
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            rows = sum(1 for _ in reader)
        result[split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "rows_before_filtering": rows, "columns": columns}
    return result


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        digest.update(str(path.relative_to(PACKAGE_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_worker(config_path, run_dir):
    """Set up workers"""
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-config", str(config_path)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", WANDB_MODE="disabled")
    env["PYTHONPATH"] = str(PACKAGE_DIR.parent) + os.pathsep + env.get("PYTHONPATH", "")
    with (run_dir / "console.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if status:
        raise RuntimeError(f"Training exited with status {status}; see {run_dir / 'console.log'}")


def collect_result(run_dir):
    results = read_yaml(run_dir / "results.yaml")
    if results.get("test") or results.get("train"):
        raise ValueError("Tuning results unexpectedly include train/hold-out evaluation")
    metrics = {name: float(results["val"][0]["val/" + name]) for name in METRICS}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError(f"Non-finite validation metrics in {run_dir}")
    summary = json.loads((run_dir / "run_summary.json").read_text())
    return {**metrics, **{k: summary[k] for k in (
        "elapsed_seconds", "epochs_completed", "best_checkpoint", "parameter_count",
    )}, "output_dir": str(run_dir)}


def save_ranking(root, candidates, records, seeds, metric, template):
    rows = []
    for name, cfg in candidates:
        runs = [r for r in records if r["candidate"] == name]
        if len(runs) != len(seeds):
            continue
        values = [r[metric] for r in runs]
        rows.append({"candidate": name, "mean": statistics.mean(values),
                     "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                     "seeds": seeds})
    # Stable sorting breaks ties by the predefined candidate order.
    rows.sort(key=lambda row: row["mean"], reverse=METRICS[metric] == "max")
    write_json(root / "ranking.json", {"metric": metric, "candidates": rows})
    if not rows:
        return
    winner = rows[0]
    best = next(cfg for name, cfg in candidates if name == winner["candidate"])
    for cfg, suffix in zip(final_configs(best, template), ("", "_diff")):
        write_yaml(root / f"best_params{suffix}.yaml", cfg)
    write_json(root / "best_params.json", {**winner, "metric": metric,
        "checkpoint_metric": best["experiment"]["monitor_metric"],
        "all_candidates_complete": len(rows) == len(candidates),
        "implementation": IMPLEMENTATION, "logit_rounding": "preserved",
        "shared_encoder_parameters": sorted(SHARED_ENCODER),
        "diffusion_head": "kept from diffusion template; baseline readout is replaced"})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["chemprop"], default="chemprop")
    parser.add_argument("--config", type=Path, default=TOX21_DIR / "chemprop.yaml")
    parser.add_argument("--diff-config", type=Path, default=TOX21_DIR / "chemprop_diff.yaml")
    parser.add_argument("--candidates", type=Path, default=TOX21_DIR / "chemprop_manual_candidates.yaml")
    parser.add_argument("--only", nargs="+", help="Run only these named candidates")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--seeds", type=int, nargs="+", help="Optional repeated tuning seeds; rank mean validation score")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto")
    parser.add_argument("--max-epochs", type=int, help="Override the config epoch budget (currently 800)")
    parser.add_argument("--es-patience", type=int, help="Override patience in validation checks (currently 15)")
    parser.add_argument("--check-val-freq", type=int, help="Validate every N epochs (currently 10)")
    parser.add_argument("--checkpoint-metric", choices=METRICS, default="pr_auc",
                        help="Checkpoint and early-stop metric; defaults to validation AP")
    parser.add_argument("--selection-metric", choices=METRICS, default="pr_auc",
                        help="Candidate ranking metric; retains your previous AP objective")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any(x is not None and x < 1 for x in (args.max_epochs, args.es_patience, args.check_val_freq)):
        parser.error("Epoch budget, validation frequency and patience must be positive")
    if args.resume and args.output_root is None:
        parser.error("--resume requires --output-root")
    args.seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Seeds must be distinct integers in [0, 2**32)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_config:
        from omegaconf import OmegaConf
        from diffpred.run import MPPExperiment
        MPPExperiment(OmegaConf.load(args.worker_config))()
        return
    base, template = training_defaults(read_yaml(args.config)), read_yaml(args.diff_config)
    if (template["model"]["hparams"]["net_type"] != "chemprop"
            or template["data"]["diffusion_params"]["diffusion_type"] is None):
        raise ValueError("--diff-config must be a Chemprop diffusion template")
    if args.max_epochs is not None:
        base["model"]["training"]["max_epochs"] = args.max_epochs
    if args.es_patience is not None:
        base["experiment"]["early_stopping_patience"] = args.es_patience
    if args.check_val_freq is not None:
        base["experiment"]["check_val_freq"] = args.check_val_freq
    if base["experiment"]["check_val_freq"] > base["model"]["training"]["max_epochs"]:
        raise ValueError("Validation frequency exceeds epoch budget; lower --check-val-freq")
    base["data"].update(load_test=False, standardize_targets=False, symmetric_binary_targets=False)
    base["experiment"].update(
        tune=True, debug=False, wandb_mode="disabled", console_out=True,
        seed=args.seeds[0], device=args.device, use_ema_on_test=False,
        monitor_metric="val/" + args.checkpoint_metric,
        monitor_mode=METRICS[args.checkpoint_metric], early_stopping=True,
        use_last_ckpt=False, save_last_ckpt=False,
    )
    directory = (args.data_dir or Path(base["data"]["data_dir"])).expanduser()
    if args.data_dir is None and not directory.is_dir() and not directory.is_absolute():
        directory = PACKAGE_DIR / directory
    base["data"]["data_dir"] = str(directory.resolve())
    candidates = build_candidates(base, read_yaml(args.candidates)["candidates"])
    if args.only:
        unknown = set(args.only) - {name for name, _ in candidates}
        if unknown:
            raise ValueError(f"Unknown candidate names: {sorted(unknown)}")
        candidates = [(name, cfg) for name, cfg in candidates if name in args.only]
    print(f"Manual candidates: {len(candidates)}; seeds: {args.seeds}; fits: {len(candidates) * len(args.seeds)}", flush=True)
    print(f"Checkpoint: validation {args.checkpoint_metric}; ranking: validation {args.selection_metric}; "
          "legacy logit rounding retained.", flush=True)
    print(f"Implementation: {IMPLEMENTATION}", flush=True)
    if args.dry_run:
        print(yaml.safe_dump({"implementation": IMPLEMENTATION, "candidates": [
            {"name": name, "config": cfg} for name, cfg in candidates]}, sort_keys=False))
        return
    dataset = data_fingerprint(Path(base["data"]["data_dir"]))
    print("CSV rows before filtering: " + ", ".join(
        f"{split}={info['rows_before_filtering']}" for split, info in dataset.items()), flush=True)
    manifest = {"version": 2, "implementation": IMPLEMENTATION, "candidates": candidates,
                "seeds": args.seeds, "selection_metric": args.selection_metric,
                "diffusion_template": template, "data": dataset, "code_sha256": code_fingerprint()}
    # Normalize tuples to lists for exact JSON resume comparisons.
    manifest = json.loads(json.dumps(manifest))
    root = output_root(args.output_root)
    if root.exists() and any(root.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Output is not empty: {root}. Use --resume or a new directory")
        if json.loads((root / "manifest.json").read_text()) != manifest:
            raise ValueError("Resume rejected: code, data, candidates or settings have changed; use a new output directory")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "manifest.json", manifest)
    print(f"Output: {root}", flush=True)
    records = []
    for index, (name, cfg) in enumerate(candidates, start=1):
        for seed in args.seeds:
            job = root / name / f"seed_{seed}"
            job.mkdir(parents=True, exist_ok=True)
            complete = job / "completed.json"
            if complete.exists():
                record = json.loads(complete.read_text())
                print(f"Resume: {name}, seed {seed} already completed", flush=True)
            else:
                # Retain interrupted attempts for diagnosis; do not reuse their checkpoints.
                attempt = job / datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
                attempt.mkdir()
                run_cfg = copy.deepcopy(cfg)
                run_cfg["experiment"].update(seed=seed, output_dir=str(attempt),
                                             exp_name=f"jku_{name}_seed{seed}")
                config_path = attempt / "requested_config.yaml"
                write_yaml(config_path, run_cfg)
                print(f"[{index}/{len(candidates)}] {name}, seed {seed}: starting", flush=True)
                run_worker(config_path, attempt)
                record = {"candidate": name, "seed": seed, **collect_result(attempt)}
                write_json(complete, record)
            records.append(record)
            with (root / "validation_scores.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            print(f"{name}, seed {seed}: AP={record['pr_auc']:.6f}, ROC-AUC={record['roc_auc']:.6f}, "
                  f"loss={record['loss']:.6f}, seconds={record['elapsed_seconds']:.1f}", flush=True)
            save_ranking(root, candidates, records, args.seeds, args.selection_metric, template)
    print(f"Finished. Selected configs: {root / 'best_params.yaml'} and {root / 'best_params_diff.yaml'}", flush=True)


if __name__ == "__main__":
    main()
