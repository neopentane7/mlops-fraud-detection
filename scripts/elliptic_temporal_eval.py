"""Temporal evaluation of the Elliptic dataset, replicating Weber et al. (2019).

The production pipeline evaluates `elliptic` on a stratified random split (for
comparability with the other profiles). That is optimistic for this dataset:
the dataset's origin paper splits *temporally* — train on time steps 1-34, test
on 35-49 — which is harder because a dark-market shutdown around step 43 shifts
the fraud dynamics.

This standalone analysis reproduces the paper's protocol so the numbers are
genuinely comparable, and prints the per-time-step recall collapse that is the
dataset's signature finding. It trains the same XGBoost configuration the
`elliptic` profile uses (165 node features), at the paper's default 0.5
threshold.

    python scripts/elliptic_temporal_eval.py    # needs Kaggle creds + xgboost
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import load_config

TRAIN_MAX_STEP = 34  # paper: train steps 1-34, test 35-49
KAGGLE_DATASET = "ellipticco/elliptic-data-set"


def assemble() -> tuple[pd.DataFrame, list[str]]:
    """Download + join Elliptic into a labelled table with time_step retained."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    tmp = Path(tempfile.mkdtemp())
    try:
        api.dataset_download_files(KAGGLE_DATASET, path=str(tmp), unzip=True)
        feats = pd.read_csv(next(tmp.rglob("elliptic_txs_features.csv")), header=None)
        n = feats.shape[1] - 2
        feature_cols = [f"f{i}" for i in range(1, n + 1)]
        feats.columns = ["txId", "time_step", *feature_cols]
        classes = pd.read_csv(next(tmp.rglob("elliptic_txs_classes.csv")))
        df = feats.merge(classes, on="txId")
        df = df[df["class"] != "unknown"].copy()
        df["y"] = (df["class"].astype(str) == "1").astype(int)  # illicit -> 1
        return df, feature_cols
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict[str, float]:
    pred = (prob >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "auprc": average_precision_score(y, prob),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Elliptic temporal evaluation.")
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Write the metrics and per-step recall to this JSON path. Without "
            "it the numbers exist only as terminal output and cannot be traced "
            "back to a run."
        ),
    )
    args = parser.parse_args()

    df, feats = assemble()
    print(
        f"labelled={len(df)} illicit_rate={df['y'].mean():.4f} "
        f"steps={int(df['time_step'].min())}-{int(df['time_step'].max())}"
    )

    train = df[df["time_step"] <= TRAIN_MAX_STEP]
    test = df[df["time_step"] > TRAIN_MAX_STEP]
    print(
        f"temporal split: train={len(train)} (steps 1-{TRAIN_MAX_STEP}) "
        f"test={len(test)} (steps {TRAIN_MAX_STEP + 1}-{int(df['time_step'].max())})"
    )

    spw = float((train["y"] == 0).sum() / max((train["y"] == 1).sum(), 1))
    # Hyperparameters come from params.yaml, not a second hardcoded copy: a
    # duplicated literal here would silently drift from the pipeline it claims
    # to reproduce, making the published temporal numbers incomparable.
    train_cfg = load_config().train
    model = xgb.XGBClassifier(
        n_estimators=train_cfg.n_estimators,
        max_depth=train_cfg.max_depth,
        learning_rate=train_cfg.learning_rate,
        scale_pos_weight=spw,  # computed from the temporal train split
        subsample=train_cfg.subsample,
        colsample_bytree=train_cfg.colsample_bytree,
        random_state=train_cfg.random_seed,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )
    print(
        f"hyperparams from params.yaml: n_estimators={train_cfg.n_estimators} "
        f"max_depth={train_cfg.max_depth} lr={train_cfg.learning_rate} "
        f"subsample={train_cfg.subsample} colsample={train_cfg.colsample_bytree} "
        f"seed={train_cfg.random_seed} scale_pos_weight={spw:.2f}"
    )
    model.fit(train[feats], train["y"])
    prob = model.predict_proba(test[feats])[:, 1]

    m = _metrics(test["y"].to_numpy(), prob, 0.5)
    print("\n=== TEMPORAL test, threshold 0.5 (paper protocol) ===")
    for k, v in m.items():
        print(f"  {k:10s} {v:.4f}")

    per_step: list[dict[str, float]] = []
    print("\n=== per-time-step recall (test period; dark-market shutdown ~43) ===")
    for step in sorted(test["time_step"].unique()):
        sub = test[test["time_step"] == step]
        if sub["y"].sum() == 0:
            continue
        pred = (model.predict_proba(sub[feats])[:, 1] >= 0.5).astype(int)
        rec = recall_score(sub["y"], pred, zero_division=0)
        prec = precision_score(sub["y"], pred, zero_division=0)
        per_step.append(
            {
                "time_step": int(step),
                "n": int(len(sub)),
                "illicit": int(sub["y"].sum()),
                "recall": float(rec),
                "precision": float(prec),
            }
        )
        print(
            f"  step {int(step):2d}: n={len(sub):5d} illicit={int(sub['y'].sum()):4d} "
            f"recall={rec:.3f} precision={prec:.3f}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "protocol": "Weber et al. (2019) temporal split",
                    "train_max_time_step": TRAIN_MAX_STEP,
                    "threshold": 0.5,
                    "n_labelled": int(len(df)),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "illicit_rate": float(df["y"].mean()),
                    "scale_pos_weight": spw,
                    "hyperparameters": {
                        "n_estimators": train_cfg.n_estimators,
                        "max_depth": train_cfg.max_depth,
                        "learning_rate": train_cfg.learning_rate,
                        "subsample": train_cfg.subsample,
                        "colsample_bytree": train_cfg.colsample_bytree,
                        "random_seed": train_cfg.random_seed,
                    },
                    "metrics": m,
                    "per_time_step": per_step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[temporal] metrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
