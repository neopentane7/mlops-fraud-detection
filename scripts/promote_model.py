"""Model Registry promotion utilities used by the CD and retrain workflows.

Promotion is a *registry stage transition*, never a code change or redeploy.
This script wraps the two transitions the pipeline needs:

* ``--compare``: compare the latest Staging challenger's holdout f1_fraud
  against the current Production champion and promote only if it wins by at
  least ``--min-delta``. Invoked by ``.github/workflows/retrain.yml``.
* ``--promote staging-to-prod``: move the latest Staging version to Production
  (archiving the previous one). This is an **operator command**, run by hand
  against a real tracking server. No workflow calls it: ``cd.yml`` promotes the
  container *image* to ``:latest`` and does not touch the model registry, since
  its MLflow server is job-scoped and gone by the release stage.
"""

from __future__ import annotations

import argparse
import json
import sys

import mlflow
from mlflow.exceptions import MlflowException

from src.config import EVAL_METRICS_PATH, REGISTERED_MODEL_NAME


def _client() -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient()


def _latest_version(stage: str) -> str | None:
    """Return the latest model version in ``stage``, or None.

    Returns None (rather than raising) when the registry is unavailable or the
    model was never registered — e.g. a file-store backend (no registry) or a
    fresh tracking server. Promotion is then a safe no-op.
    """
    try:
        versions = _client().get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage])
    except MlflowException as exc:
        print(f"[promote] registry unavailable / model not registered: {exc}")
        return None
    return versions[0].version if versions else None


def promote_staging_to_production() -> int:
    """Transition the latest Staging version to Production, archiving the old."""
    client = _client()
    version = _latest_version("Staging")
    if version is None:
        # Nothing to promote (no registry, or no Staging version) — a no-op,
        # not a failure, so it never breaks the CD/retrain workflow.
        print("[promote] No Staging version to promote; skipping (no-op).")
        return 0
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[promote] v{version} -> Production (previous Production archived)")
    return 0


def _champion_holdout_f1(run_id: str) -> float | None:
    """Read the champion's **holdout** f1_fraud from its MLflow run.

    ``train.py`` logs *validation* metrics as ``f1_fraud``; ``evaluate.py
    --stage holdout`` back-annotates the same run with ``holdout_f1_fraud``.
    Only the latter is comparable to the challenger's holdout number, so a run
    that predates that back-annotation returns ``None`` rather than silently
    handing back a validation score measured on a different split.
    """
    metrics = _client().get_run(run_id).data.metrics
    if "holdout_f1_fraud" in metrics:
        return float(metrics["holdout_f1_fraud"])
    return None


def compare_and_promote(min_delta: float) -> int:
    """Promote Staging only if it beats Production f1_fraud by >= ``min_delta``.

    Both sides of the comparison are **holdout** f1_fraud, measured on the same
    test split. Comparing the challenger's holdout score against the champion's
    validation score (which is what the run's bare ``f1_fraud`` records) would
    compare two different splits and bias the gate against promotion.
    """
    if not EVAL_METRICS_PATH.exists():
        print(f"[promote] No holdout metrics at {EVAL_METRICS_PATH}; skipping (no-op).")
        return 0
    challenger_f1 = float(json.loads(EVAL_METRICS_PATH.read_text())["f1_fraud"])
    prod_version = _latest_version("Production")
    if prod_version is None:
        print("[promote] No Production model; promoting challenger unconditionally.")
        return promote_staging_to_production()

    run_id = _client().get_model_version(REGISTERED_MODEL_NAME, prod_version).run_id
    prod_f1 = _champion_holdout_f1(run_id)
    if prod_f1 is None:
        print(
            f"[promote] Champion run {run_id} has no 'holdout_f1_fraud' metric "
            "(trained before holdout back-annotation). Refusing to compare a "
            "holdout score against a validation score; keeping current "
            "Production. Re-run `evaluate --stage holdout` for that model to "
            "make it comparable."
        )
        return 0
    delta = challenger_f1 - prod_f1
    print(
        f"[promote] challenger f1={challenger_f1:.4f} prod f1={prod_f1:.4f} "
        f"delta={delta:+.4f}"
    )
    if delta >= min_delta:
        return promote_staging_to_production()
    print(
        f"[promote] Challenger did not win by >= {min_delta}; "
        "keeping current Production."
    )
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Promote registered models.")
    parser.add_argument("--promote", choices=["staging-to-prod"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--min-delta", type=float, default=0.02)
    args = parser.parse_args()

    if args.compare:
        code = compare_and_promote(args.min_delta)
    elif args.promote == "staging-to-prod":
        code = promote_staging_to_production()
    else:
        parser.error("Specify --promote staging-to-prod or --compare")
    return code


if __name__ == "__main__":
    sys.exit(main())
