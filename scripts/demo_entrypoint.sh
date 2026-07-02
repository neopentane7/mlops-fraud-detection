#!/bin/sh
# Self-contained demo startup: serve the REAL credit-card fraud model that was
# trained and baked into the image at build time (see Dockerfile.demo). There is
# no training, no Kaggle data, no tracking server and no secrets at runtime — the
# model, scaler and threshold are already on disk, so startup is fast and the
# container fits a 512 MB free tier.
set -e

export MLOPS_DATASET=creditcard
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"   # baked local store; runs:/ resolves here
export MODEL_NAME=fraud-detector
export PYTHONUTF8=1
# The demo is intentionally public with no API key, and binds 0.0.0.0, so give
# the open /predict + /predict/batch endpoints a default per-IP rate limit to
# blunt abuse of a hosted deployment. Override at deploy time if needed.
export RATE_LIMIT_PER_MINUTE="${RATE_LIMIT_PER_MINUTE:-60}"

# Point the API at the baked model artifact (bypasses the registry).
MODEL_URI=$(python -c "import json; print(json.load(open('models/creditcard/run_info.json'))['model_uri'])")
export MODEL_URI
echo "[demo] serving baked creditcard model MODEL_URI=$MODEL_URI on port ${PORT:-7860}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
