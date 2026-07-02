"""Generate a comparable core figure set for the ACTIVE dataset into an outdir.

Dataset-agnostic companion to ``generate_figures.py`` (which also produces the
creditcard-specific ``Amount``/``scale_pos_weight``/seed-variance charts). Run
once per dataset via ``MLOPS_DATASET``; it trains the pipeline's base model
(applying the same per-dataset ``scale_pos_weight``/``min_recall`` overrides) and
writes six comparable charts, then optionally dumps the test predictions so
``generate_comparison.py`` can overlay all datasets on one axis.

Usage:
    MLOPS_DATASET=elliptic python scripts/generate_dataset_figures.py \
        --outdir docs/images/elliptic --dump /tmp/elliptic.npz
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    ACTIVE_DATASET,
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TRAIN_OVERRIDES,
    Config,
    load_config,
)

plt.rcParams.update(
    {"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3}
)
BLUE, RED = "#2c6fbb", "#c0392b"

# Human-readable name for the positive class, per dataset.
POS_LABEL = {
    "creditcard": "fraud",
    "cc-default": "default",
    "elliptic": "illicit",
}.get(ACTIVE_DATASET, "positive")


def resolved_spw_and_recall(
    y_train: np.ndarray, cfg: Config
) -> tuple[float, float]:
    """Mirror train.apply_train_overrides for scale_pos_weight/min_recall only.

    Kept inline (rather than importing train.py) so this figure script has no
    MLflow dependency.
    """
    spw = TRAIN_OVERRIDES.get("scale_pos_weight", cfg.train.scale_pos_weight)
    if spw == "auto":
        pos = max(int((y_train == 1).sum()), 1)
        spw = round(float((y_train == 0).sum()) / pos, 2)
    min_recall = float(TRAIN_OVERRIDES.get("min_recall", cfg.train.min_recall))
    return float(spw), min_recall


def recall_first_threshold(
    y_val: np.ndarray, p_val: np.ndarray, min_recall: float
) -> float:
    """Recall-first boundary threshold — mirrors train.find_optimal_threshold."""
    if int(np.asarray(y_val).sum()) == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if thr.size == 0:
        return 0.5
    prec, rec = prec[:-1], rec[:-1]
    meets = rec >= min_recall
    if meets.any():
        return float(thr[int(np.where(meets)[0][-1])])
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thr[int(np.argmax(f1))]) if f1.size else 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, help="directory for the PNGs")
    parser.add_argument("--dump", default="", help="optional .npz of test preds")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    feats = list(FEATURE_COLUMNS)

    raw = pd.read_csv(RAW_DATA_PATH)
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    spw, min_recall = resolved_spw_and_recall(train[TARGET_COLUMN].to_numpy(), cfg)
    model = xgb.XGBClassifier(
        n_estimators=cfg.train.n_estimators,
        max_depth=cfg.train.max_depth,
        learning_rate=cfg.train.learning_rate,
        scale_pos_weight=spw,
        subsample=cfg.train.subsample,
        colsample_bytree=cfg.train.colsample_bytree,
        random_state=cfg.train.random_seed,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(train[feats], train[TARGET_COLUMN])
    vp = model.predict_proba(val[feats])[:, 1]
    tp = model.predict_proba(test[feats])[:, 1]
    yv, yt = val[TARGET_COLUMN].to_numpy(), test[TARGET_COLUMN].to_numpy()
    chosen = recall_first_threshold(yv, vp, min_recall)

    auprc = float(average_precision_score(yt, tp))
    auc = float(roc_auc_score(yt, tp))
    title = ACTIVE_DATASET

    # ---- class imbalance -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = raw[TARGET_COLUMN].value_counts().sort_index()
    labels = ["negative (0)", f"{POS_LABEL} (1)"]
    bars = ax.bar(labels, counts.values, color=[BLUE, RED])
    ax.set_yscale("log")
    ax.set_ylabel("count (log scale)")
    neg, pos = counts.iloc[0], counts.iloc[1]
    ratio = neg / pos
    ax.set_title(f"{title} — class imbalance {neg:,} : {pos:,} ({ratio:.1f}:1)")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "class_imbalance.png")
    plt.close(fig)

    # ---- PR curve --------------------------------------------------------
    prec_t, rec_t, thr_t = precision_recall_curve(yt, tp)
    op = int(np.argmin(np.abs(thr_t - chosen)))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rec_t, prec_t, color=BLUE, label=f"PR curve (AUPRC={auprc:.3f})")
    ax.scatter(rec_t[op], prec_t[op], color=RED, zorder=5, s=60,
               label=f"operating point @ thr={chosen:.3f}")
    ax.set_xlabel(f"Recall ({POS_LABEL})")
    ax.set_ylabel(f"Precision ({POS_LABEL})")
    ax.set_title(f"{title} — Precision-Recall (test)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png")
    plt.close(fig)

    # ---- ROC curve -------------------------------------------------------
    fpr, tpr, _ = roc_curve(yt, tp)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color=BLUE, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title} — ROC (test)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png")
    plt.close(fig)

    # ---- threshold trade-off (validation) --------------------------------
    grid = np.linspace(0.01, 0.99, 99)
    prec_g, rec_g, f1_g = [], [], []
    for t in grid:
        pred = (vp >= t).astype(int)
        tp_ = int(((pred == 1) & (yv == 1)).sum())
        fp_ = int(((pred == 1) & (yv == 0)).sum())
        fn_ = int(((pred == 0) & (yv == 1)).sum())
        p = tp_ / (tp_ + fp_ + 1e-9)
        r = tp_ / (tp_ + fn_ + 1e-9)
        prec_g.append(p)
        rec_g.append(r)
        f1_g.append(2 * p * r / (p + r + 1e-9))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, prec_g, color=BLUE, label="precision")
    ax.plot(grid, rec_g, color=RED, label="recall")
    ax.plot(grid, f1_g, color="green", label="F1")
    ax.axhline(
        min_recall, ls=":", color=RED, alpha=0.7, label=f"recall floor={min_recall:g}"
    )
    ax.axvline(chosen, ls="--", color="black", label=f"chosen thr={chosen:.3f}")
    f1_thr = float(grid[int(np.argmax(f1_g))])
    ax.axvline(
        f1_thr, ls="--", color="grey", alpha=0.7, label=f"max-F1 thr={f1_thr:.3f}"
    )
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score (validation)")
    ax.set_title(f"{title} — threshold trade-off: recall-first vs max-F1")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "threshold_tradeoff.png")
    plt.close(fig)

    # ---- confusion matrix ------------------------------------------------
    cm = confusion_matrix(yt, (tp >= chosen).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred neg", f"pred {POS_LABEL}"])
    ax.set_yticks([0, 1], ["true neg", f"true {POS_LABEL}"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title(f"{title} — confusion matrix (test) @ thr={chosen:.3f}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png")
    plt.close(fig)

    # ---- feature importance ----------------------------------------------
    gain = model.get_booster().get_score(importance_type="gain")
    imp = pd.Series(gain).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("gain")
    ax.set_title(f"{title} — top-15 feature importance (gain)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png")
    plt.close(fig)

    print(f"[{ACTIVE_DATASET}] figures -> {out}  (spw={spw:g} thr={chosen:.4f} "
          f"auc={auc:.4f} auprc={auprc:.4f})")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.dump,
            y=yt,
            p=tp,
            threshold=chosen,
            name=ACTIVE_DATASET,
            pos_label=POS_LABEL,
        )
        print(f"[{ACTIVE_DATASET}] test preds dumped -> {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
