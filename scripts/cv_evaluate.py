"""Repeated stratified K-fold evaluation — the evidence behind the gate targets.

The benchmark targets in ``params.yaml`` are calibrated from cross-validated
runs, not from a single split. This script is what produces that evidence, and
it writes a machine-readable artifact so the numbers quoted in the README and
model card can be traced to a run instead of a terminal transcript.

Why it exists at all: a single holdout split of this dataset contains only ~71
frauds, so the operating point (precision/recall/F1) is high variance while the
ranking metrics (ROC-AUC, AUPRC) are stable. Gating on one split therefore risks
both false failures and false passes. Reporting mean +/- std across folds shows
which metrics are trustworthy enough to gate on — and
:func:`check_cv_benchmarks` can gate on the fold mean directly.

Methodology, chosen to avoid the leakage a naive CV loop introduces:

* the scaler is fit on each fold's **training** portion only, never on the full
  dataset, so no test-fold statistics leak into training;
* the decision threshold is tuned on a **validation slice carved out of the
  training portion**, never on the test fold, mirroring what the pipeline does;
* duplicates are dropped **before** splitting, exactly as ``preprocess.py`` does,
  so the same row cannot land in both train and test.

Each seed perturbs **both** the data partition and the model's own RNG, so the
spread reported here is what a genuine re-run of the pipeline would experience,
not split variance alone.

Everything reuses the pipeline's own functions (``build_model``,
``find_optimal_threshold``, ``compute_metrics``), so this cannot silently drift
away from what the pipeline actually does.

    python scripts/cv_evaluate.py                      # 5 folds x 5 seeds
    python scripts/cv_evaluate.py --folds 5 --seeds 42 # one quick pass
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import (
    ACTIVE_DATASET,
    BENCHMARK_TARGETS,
    FEATURE_COLUMNS,
    METRICS_DIR,
    RAW_DATA_PATH,
    SCALED_COLUMNS,
    TARGET_COLUMN,
    Config,
    load_config,
)
from src.models.evaluate import compute_metrics
from src.models.train import build_model, find_optimal_threshold

# Metrics whose stability across folds is the point of the exercise.
REPORTED = (
    "roc_auc",
    "avg_precision",
    "f1_fraud",
    "precision_fraud",
    "recall_fraud",
)
DEFAULT_SEEDS = (42, 7, 21, 123, 1)
# Fraction of each fold's training portion held out to tune the threshold.
VAL_FRACTION = 0.1765  # ~= 0.15/0.85, matching the pipeline's 70/15/15 shape


def load_deduplicated() -> pd.DataFrame:
    """Load the raw dataset and drop exact duplicates, as preprocess.py does."""
    frame = pd.read_csv(RAW_DATA_PATH)
    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    print(
        f"[cv] {RAW_DATA_PATH.name}: {before:,} rows -> {len(frame):,} after dedup "
        f"({int(frame[TARGET_COLUMN].sum())} positives)"
    )
    return frame


def evaluate_fold(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config, seed: int
) -> dict[str, float]:
    """Fit, tune a threshold, and score one fold without leaking the test split."""
    features = list(FEATURE_COLUMNS)
    scaled = list(SCALED_COLUMNS)

    # Threshold-tuning slice comes out of TRAIN, never out of the test fold.
    fit_df, val_df = train_test_split(
        train_df,
        test_size=VAL_FRACTION,
        random_state=seed,
        stratify=train_df[TARGET_COLUMN],
    )
    fit_df, val_df, test_df = fit_df.copy(), val_df.copy(), test_df.copy()

    if scaled:
        scaler = RobustScaler().fit(fit_df[scaled])
        for part in (fit_df, val_df, test_df):
            part[scaled] = scaler.transform(part[scaled])

    # Vary the model's own RNG with the seed as well, not just the partition.
    # Holding random_state fixed would make every "seed" reuse identical
    # subsample/colsample draws, measuring split variance only. A real re-run
    # perturbs both, so both are perturbed here.
    model = build_model(replace(cfg.train, random_seed=seed))
    model.fit(fit_df[features], fit_df[TARGET_COLUMN])

    val_probs = model.predict_proba(val_df[features])[:, 1]
    threshold = find_optimal_threshold(
        model,
        val_df[features],
        val_df[TARGET_COLUMN],
        min_recall=cfg.train.min_recall,
        threshold_strategy=cfg.train.threshold_strategy,
        cost_fn=cfg.train.cost_fn,
        cost_fp=cfg.train.cost_fp,
        probs=val_probs,
    )
    test_probs = model.predict_proba(test_df[features])[:, 1]
    return compute_metrics(test_df[TARGET_COLUMN], test_probs, threshold)


def run_cv(
    frame: pd.DataFrame, folds: int, seeds: tuple[int, ...], cfg: Config
) -> list[dict[str, float]]:
    """Run repeated stratified K-fold, returning one metric dict per fold."""
    features_and_target = frame[list(FEATURE_COLUMNS) + [TARGET_COLUMN]]
    y = features_and_target[TARGET_COLUMN]
    results: list[dict[str, float]] = []

    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(features_and_target, y)
        for k, (train_idx, test_idx) in enumerate(splits, 1):
            started = time.perf_counter()
            metrics = evaluate_fold(
                features_and_target.iloc[train_idx],
                features_and_target.iloc[test_idx],
                cfg,
                seed,
            )
            metrics["seed"] = float(seed)
            metrics["fold"] = float(k)
            results.append(metrics)
            print(
                f"[cv] seed={seed:<4} fold {k}/{folds}  "
                + "  ".join(f"{m}={metrics[m]:.4f}" for m in REPORTED)
                + f"  ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    return results


def summarise(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metrics into mean/std/min/max."""
    summary: dict[str, dict[str, float]] = {}
    for name in REPORTED:
        values = [r[name] for r in results]
        summary[name] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summary


def check_cv_benchmarks(
    summary: dict[str, dict[str, float]], targets: dict[str, float]
) -> list[str]:
    """Gate on the fold **mean** rather than a single noisy split.

    This is the CV-based gating the project recommends as the more robust
    successor to single-split gating: a target is missed only if the average
    across every fold misses it, so one unlucky split cannot fail the build and,
    equally, cannot sneak a regression through.
    """
    failures: list[str] = []
    for name, target in targets.items():
        stats = summary.get(name)
        if stats is None:
            failures.append(f"{name}=<not evaluated> < target {target:.2f}")
        elif stats["mean"] < target:
            failures.append(
                f"{name} mean={stats['mean']:.4f} (+/-{stats['std']:.4f}) "
                f"< target {target:.2f}"
            )
    return failures


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds, e.g. 42,7,21",
    )
    parser.add_argument("--out", type=Path, default=METRICS_DIR / "cv_metrics.json")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if a fold-mean misses its benchmark target",
    )
    args = parser.parse_args()

    cfg = load_config()
    frame = load_deduplicated()
    n_models = args.folds * len(args.seeds)
    print(
        f"[cv] dataset={ACTIVE_DATASET} folds={args.folds} seeds={list(args.seeds)} "
        f"-> {n_models} models"
    )

    started = time.perf_counter()
    results = run_cv(frame, args.folds, args.seeds, cfg)
    summary = summarise(results)
    elapsed = time.perf_counter() - started

    print(f"\n[cv] {n_models} models in {elapsed:.0f}s\n")
    print(f"{'metric':<18}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'target':>9}  ")
    print("-" * 72)
    for name in REPORTED:
        st = summary[name]
        target = BENCHMARK_TARGETS.get(name)
        tgt = f"{target:.2f}" if target is not None else "-"
        print(
            f"{name:<18}{st['mean']:>9.4f}{st['std']:>9.4f}"
            f"{st['min']:>9.4f}{st['max']:>9.4f}{tgt:>9}  "
        )

    payload = {
        "dataset": ACTIVE_DATASET,
        "n_folds": args.folds,
        "seeds": list(args.seeds),
        "n_models": n_models,
        "n_rows": int(len(frame)),
        "n_positives": int(frame[TARGET_COLUMN].sum()),
        "elapsed_seconds": round(elapsed, 1),
        "summary": summary,
        "benchmark_targets": dict(BENCHMARK_TARGETS),
        "folds": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[cv] summary -> {args.out}")

    failures = check_cv_benchmarks(summary, dict(BENCHMARK_TARGETS))
    if failures:
        print("[cv] fold-mean below target:")
        for failure in failures:
            print(f"  - {failure}")
        if args.gate:
            return 1
    else:
        print("[cv] every benchmark target is met by the fold mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
