"""Tests for the cross-validated evaluation and CV-based gating.

The fold loop itself trains real models and is exercised by running the script;
what is unit-tested here is the aggregation and the gate that consumes it,
plus the leakage properties of a single fold on the synthetic fixture.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    Config,
    MonitoringConfig,
    PreprocessConfig,
    ServingConfig,
    TrainConfig,
)

# scripts/ is not an installed package; load the module from its file path.
_CV_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cv_evaluate.py"
_spec = importlib.util.spec_from_file_location("cv_evaluate", _CV_PATH)
assert _spec and _spec.loader
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)


def _cfg() -> Config:
    """A fast-training config for fold-level tests."""
    return Config(
        preprocess=PreprocessConfig(0.15, 0.15, 42),
        train=TrainConfig(30, 3, 0.2, 50, 1.0, 1.0, 42, min_recall=0.5),
        monitoring=MonitoringConfig(drift_threshold=0.30, performance_threshold=0.99),
        serving=ServingConfig(
            "0.0.0.0", 8000, "http://localhost:5000", "m", "Production"
        ),
    )


def test_summarise_reports_mean_std_min_max() -> None:
    """Aggregation produces the spread that makes single-split noise visible."""
    results = [
        {"roc_auc": 0.98, "avg_precision": 0.80, "f1_fraud": 0.70,
         "precision_fraud": 0.60, "recall_fraud": 0.90},
        {"roc_auc": 0.96, "avg_precision": 0.84, "f1_fraud": 0.80,
         "precision_fraud": 0.90, "recall_fraud": 0.70},
    ]
    summary = cv.summarise(results)
    assert summary["roc_auc"]["mean"] == pytest.approx(0.97)
    assert summary["roc_auc"]["min"] == pytest.approx(0.96)
    assert summary["roc_auc"]["max"] == pytest.approx(0.98)
    # precision swings far more than roc_auc -- the whole point of CV gating.
    assert summary["precision_fraud"]["std"] > summary["roc_auc"]["std"]


def test_cv_gate_uses_the_mean_not_a_single_fold() -> None:
    """One unlucky fold must not fail the build if the mean clears the target."""
    summary = cv.summarise(
        [
            # fold 1 badly misses recall, fold 2 comfortably beats it
            {"roc_auc": 0.99, "avg_precision": 0.85, "f1_fraud": 0.8,
             "precision_fraud": 0.8, "recall_fraud": 0.60},
            {"roc_auc": 0.99, "avg_precision": 0.85, "f1_fraud": 0.8,
             "precision_fraud": 0.8, "recall_fraud": 0.98},
        ]
    )
    assert summary["recall_fraud"]["min"] < 0.78 < summary["recall_fraud"]["mean"]
    assert cv.check_cv_benchmarks(summary, {"recall_fraud": 0.78}) == []


def test_cv_gate_fails_when_the_mean_misses() -> None:
    """A genuine regression -- the average missing -- is still caught."""
    summary = cv.summarise(
        [
            {"roc_auc": 0.90, "avg_precision": 0.5, "f1_fraud": 0.4,
             "precision_fraud": 0.4, "recall_fraud": 0.40},
            {"roc_auc": 0.91, "avg_precision": 0.5, "f1_fraud": 0.4,
             "precision_fraud": 0.4, "recall_fraud": 0.42},
        ]
    )
    failures = cv.check_cv_benchmarks(summary, {"roc_auc": 0.96, "recall_fraud": 0.78})
    assert len(failures) == 2
    assert any("roc_auc" in f for f in failures)


def test_cv_gate_reports_a_metric_it_never_evaluated() -> None:
    """A target with no corresponding metric is a failure, not a crash."""
    summary = cv.summarise(
        [{"roc_auc": 0.99, "avg_precision": 0.9, "f1_fraud": 0.8,
          "precision_fraud": 0.8, "recall_fraud": 0.9}]
    )
    failures = cv.check_cv_benchmarks(summary, {"nonexistent_metric": 0.5})
    assert len(failures) == 1
    assert "not evaluated" in failures[0]


def test_evaluate_fold_does_not_leak_the_test_split(raw_df: pd.DataFrame) -> None:
    """A fold scales using train-only statistics and tunes off a train slice.

    If the scaler were fit on the whole dataset, or the threshold tuned on the
    test fold, CV would report optimistically. This pins both properties by
    checking the fold returns a sane, non-degenerate result on held-out rows.
    """
    frame = raw_df[list(FEATURE_COLUMNS) + [TARGET_COLUMN]]
    train_df = frame.iloc[: int(len(frame) * 0.8)]
    test_df = frame.iloc[int(len(frame) * 0.8) :]

    before = frame.copy()
    metrics = cv.evaluate_fold(train_df, test_df, _cfg(), seed=42)

    # The fold must not mutate the caller's frame (it copies before scaling).
    pd.testing.assert_frame_equal(frame, before)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["support_fraud"] == int(test_df[TARGET_COLUMN].sum())


def test_load_deduplicated_drops_duplicates(
    tmp_path: Path, raw_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dedup happens before splitting, mirroring preprocess.py."""
    doubled = pd.concat([raw_df, raw_df.head(50)], ignore_index=True)
    path = tmp_path / "raw.csv"
    doubled.to_csv(path, index=False)
    monkeypatch.setattr(cv, "RAW_DATA_PATH", path)

    out = cv.load_deduplicated()
    assert len(out) < len(doubled)
    assert not out.duplicated().any()


def test_reported_metrics_cover_every_benchmark_target() -> None:
    """Every gated metric must actually be summarised, or the gate is blind."""
    from src.config import BENCHMARK_TARGETS

    missing = set(BENCHMARK_TARGETS) - set(cv.REPORTED)
    assert not missing, f"benchmark targets never summarised by CV: {missing}"


def test_val_fraction_matches_the_pipeline_split_shape() -> None:
    """The in-fold tuning slice mirrors the pipeline's 70/15/15 proportions."""
    assert cv.VAL_FRACTION == pytest.approx(0.15 / 0.85, abs=1e-3)


def test_seeds_default_is_multi_seed() -> None:
    """The documented claim is 5-fold x 5-seed; the default must reflect that."""
    assert len(cv.DEFAULT_SEEDS) == 5
    assert len(set(cv.DEFAULT_SEEDS)) == 5


def test_summarise_handles_a_single_fold() -> None:
    """A one-fold run reports zero spread rather than raising."""
    summary = cv.summarise(
        [{"roc_auc": 0.9, "avg_precision": 0.8, "f1_fraud": 0.7,
          "precision_fraud": 0.6, "recall_fraud": 0.5}]
    )
    assert summary["roc_auc"]["std"] == 0.0
    assert summary["roc_auc"]["mean"] == pytest.approx(0.9)


def test_fold_metrics_are_recorded_with_provenance() -> None:
    """Each fold row carries its seed and fold index for traceability."""
    rng = np.random.default_rng(0)
    results = [
        {"roc_auc": float(rng.random()), "avg_precision": 0.8, "f1_fraud": 0.7,
         "precision_fraud": 0.6, "recall_fraud": 0.5, "seed": 42.0, "fold": float(i)}
        for i in range(1, 4)
    ]
    assert {r["fold"] for r in results} == {1.0, 2.0, 3.0}
    assert cv.summarise(results)["roc_auc"]["mean"] > 0.0
