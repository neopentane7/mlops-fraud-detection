"""Overlay all datasets on shared axes for a like-for-like visual comparison.

Consumes the ``.npz`` test-prediction dumps written by
``generate_dataset_figures.py`` and emits three comparison charts (ROC overlay,
PR overlay, grouped metric bars) into an output directory.

Usage:
    python scripts/generate_comparison.py \
        --dumps /tmp/creditcard.npz /tmp/cc-default.npz /tmp/elliptic.npz \
        --outdir docs/images/comparison
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

plt.rcParams.update(
    {"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3}
)
# One stable colour per dataset.
COLORS = {"creditcard": "#2c6fbb", "cc-default": "#e67e22", "elliptic": "#27ae60"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in args.dumps:
        z = np.load(d, allow_pickle=True)
        runs.append(
            {
                "name": str(z["name"]),
                "pos": str(z["pos_label"]),
                "y": z["y"].astype(int),
                "p": z["p"].astype(float),
                "thr": float(z["threshold"]),
            }
        )

    def color(name: str) -> str:
        return COLORS.get(name, "#7f8c8d")

    # ---- ROC overlay -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in runs:
        fpr, tpr, _ = roc_curve(r["y"], r["p"])
        auc = roc_auc_score(r["y"], r["p"])
        ax.plot(fpr, tpr, color=color(r["name"]), label=f"{r['name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — all datasets (test)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "roc_overlay.png")
    plt.close(fig)

    # ---- PR overlay ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in runs:
        prec, rec, _ = precision_recall_curve(r["y"], r["p"])
        auprc = average_precision_score(r["y"], r["p"])
        ax.plot(
            rec, prec, color=color(r["name"]), label=f"{r['name']} (AUPRC={auprc:.3f})"
        )
    ax.set_xlabel("Recall (positive class)")
    ax.set_ylabel("Precision (positive class)")
    ax.set_title("Precision-Recall — all datasets (test)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "pr_overlay.png")
    plt.close(fig)

    # ---- grouped metric bars --------------------------------------------
    metric_names = ["ROC-AUC", "AUPRC", "F1", "Recall", "Precision"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    n = len(runs)
    width = 0.8 / n
    x = np.arange(len(metric_names))
    for i, r in enumerate(runs):
        pred = (r["p"] >= r["thr"]).astype(int)
        vals = [
            roc_auc_score(r["y"], r["p"]),
            average_precision_score(r["y"], r["p"]),
            f1_score(r["y"], pred, zero_division=0),
            recall_score(r["y"], pred, zero_division=0),
            precision_score(r["y"], pred, zero_division=0),
        ]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=color(r["name"]), label=r["name"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (test)")
    ax.set_title("Model performance across all three datasets")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "metrics_bar.png")
    plt.close(fig)

    print(f"comparison figures -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
