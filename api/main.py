"""FastAPI application serving real-time fraud predictions.

Operational design:

* The model is loaded once at startup from the MLflow Model Registry via the
  ``lifespan`` context manager (the modern FastAPI pattern, not the deprecated
  ``@app.on_event``). The API never references a model file path — deploying a
  new model is a registry stage transition, not a code change.
* Startup is **all-or-nothing**: the model, threshold and scaler are loaded into
  locals and only published to ``STATE`` once every one of them succeeded. A
  partial failure therefore reports ``degraded`` health and returns HTTP 503
  from scoring endpoints, rather than reporting "healthy" and then raising a 500
  on the first request (a liveness probe can distinguish "process alive" from
  "not ready to serve").
* Every response carries an ``X-Request-ID`` header — including the 401/429
  rejections, because the request-ID middleware is registered *last* and so runs
  outermost (Starlette applies HTTP middleware in reverse registration order).
* Every prediction is appended to a JSONL audit log (input hash, output,
  latency). The write is handed to a worker thread so it never blocks the event
  loop, and a whole batch is written with a single open.
* ``/metrics`` exposes Prometheus-compatible metrics via the instrumentator.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import pickle
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, TypeAlias

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionFeatures,
)
from src.config import FEATURE_COLUMNS, SCALED_COLUMNS, SCALER_PATH
from src.models.predict import load_threshold

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# The loaded MLflow pyfunc model is intentionally opaque (no public stub); the
# ASGI middleware callable has a fixed request/response signature.
Model: TypeAlias = Any
Scaler: TypeAlias = Any  # fitted sklearn transformer (untyped third-party object)
CallNext: TypeAlias = Callable[[Request], Awaitable[Response]]

# --- Mutable serving state populated at startup -----------------------------
STATE: dict[str, Any] = {"model": None, "threshold": None, "scaler": None, "info": {}}


def _load_scaler(path: Path = SCALER_PATH) -> Scaler | None:
    """Load the fitted feature scaler used to match training-time preprocessing.

    The model was trained on scaled ``Time``/``Amount``; the API receives raw
    request values, so the *same* fitted scaler must be applied here to avoid
    train/serve skew.

    If the active profile declares scaled columns, a missing or unreadable
    scaler is treated as a **startup failure** rather than being swallowed:
    scoring raw values against a model trained on scaled ones produces
    confidently wrong probabilities, which is far worse than refusing to serve.
    ``None`` is returned only when the profile scales nothing.

    Raises:
        RuntimeError: If the profile has scaled columns but the scaler cannot
            be loaded (caught by :func:`lifespan`, which then serves degraded).
    """
    if not SCALED_COLUMNS:
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - any failure here must degrade
        raise RuntimeError(
            f"Scaler required for scaled columns {list(SCALED_COLUMNS)} but "
            f"could not be loaded from {path}: {exc}"
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the production model + threshold on startup; release on shutdown."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # MODEL_URI loads an explicit model (a local dir or runs:/...), bypassing the
    # registry — used by the self-contained demo deployment that has no server.
    model_uri = os.getenv("MODEL_URI", "")
    model_name = os.getenv("MODEL_NAME", "fraud-detector")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        # Load everything into locals FIRST. Publishing to STATE incrementally
        # would let a later failure (e.g. a missing threshold.json) leave
        # STATE["model"] set while STATE["threshold"] stayed None — /health would
        # report "healthy" and the first /predict would raise a 500 comparing a
        # float to None. Commit all three together or none of them.
        if model_uri:
            model = mlflow.pyfunc.load_model(model_uri)
            version = "baked"
        else:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
            version = _resolve_version(model_name, model_stage)
        threshold = load_threshold()
        scaler = _load_scaler()

        STATE["model"] = model
        STATE["threshold"] = threshold
        STATE["scaler"] = scaler
        STATE["info"] = {"name": model_name, "stage": model_stage, "version": version}
        logger.info(
            "Loaded model %s (v%s)", model_uri or f"{model_name}/{model_stage}", version
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, report via /health
        logger.error("Model load failed; serving in degraded mode: %s", exc)
        STATE["model"] = None
        STATE["threshold"] = None
        STATE["scaler"] = None
        STATE["info"] = {"name": model_name, "stage": model_stage, "version": "unknown"}
    yield
    STATE["model"] = None
    STATE["threshold"] = None
    STATE["scaler"] = None


def _resolve_version(model_name: str, model_stage: str) -> str:
    """Best-effort lookup of the concrete version behind a stage alias."""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        return str(versions[0].version) if versions else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


app = FastAPI(title="Fraud Detector API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


# Per-IP request timestamps for the in-memory rate limiter. The map is bounded:
# once it exceeds _RATE_BUCKET_MAX_IPS, every IP whose window has fully expired
# is evicted, so a long-lived process cannot grow without limit under IP-diverse
# traffic. Still per-process — front it with Redis (or a gateway) for
# multi-replica deployments.
_RATE_BUCKETS: dict[str, list[float]] = {}
_RATE_BUCKET_MAX_IPS = 10_000
_RATE_WINDOW_S = 60.0
# The limiter is a read-modify-write straddling an await point, so it needs a
# lock: without one, two concurrent requests from the same IP can both observe a
# below-limit count and both be admitted.
_rate_lock = asyncio.Lock()

# Monotonic clock, not wall clock: the window must not be skewed by NTP steps.


async def _allow_request(client_ip: str, limit: int) -> bool:
    """Record a request for ``client_ip``; return False if it exceeds ``limit``."""
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_S
    async with _rate_lock:
        if len(_RATE_BUCKETS) > _RATE_BUCKET_MAX_IPS:
            stale = [
                ip
                for ip, stamps in _RATE_BUCKETS.items()
                if not stamps or stamps[-1] <= cutoff
            ]
            for ip in stale:
                del _RATE_BUCKETS[ip]
        recent = [t for t in _RATE_BUCKETS.get(client_ip, []) if t > cutoff]
        if len(recent) >= limit:
            _RATE_BUCKETS[client_ip] = recent
            return False
        recent.append(now)
        _RATE_BUCKETS[client_ip] = recent
        return True


# NOTE ON ORDERING: Starlette applies HTTP middleware in REVERSE registration
# order, so the LAST one registered runs OUTERMOST. `add_request_id` is
# therefore defined after `auth_and_rate_limit` on purpose — it must wrap the
# 401/429 short-circuits below so that *every* response carries an
# X-Request-ID, including rejected ones. Do not reorder these two.


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next: CallNext) -> Response:
    """Optional API-key auth + per-IP rate limiting on the scoring endpoints.

    Both are off by default (no ``API_KEY`` / ``RATE_LIMIT_PER_MINUTE=0``) so
    local and CI runs need no credentials; set the env vars to enable them in
    production.
    """
    if request.url.path.startswith("/predict"):
        api_key = os.getenv("API_KEY", "")
        if api_key:
            provided = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(provided, api_key):
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid or missing API key"}
                )
        limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))
        if limit > 0:
            client_ip = request.client.host if request.client else "unknown"
            if not await _allow_request(client_ip, limit):
                return JSONResponse(
                    status_code=429, content={"detail": "Rate limit exceeded"}
                )
    return await call_next(request)


@app.middleware("http")
async def add_request_id(request: Request, call_next: CallNext) -> Response:
    """Set one X-Request-ID on request state, the response, and the audit log."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _is_ready() -> bool:
    """True when every artifact needed to serve a score is loaded."""
    return STATE["model"] is not None and STATE["threshold"] is not None


def get_model() -> Model:
    """Return the loaded model or raise HTTP 503 if the app is not ready.

    Checks the threshold too, not just the model: scoring needs both, and a
    503 is the correct answer for "not ready" (a 500 would signal a bug).
    """
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return STATE["model"]


# Serialises audit-log appends within this process. Separate uvicorn workers
# still share the file; each record is written with a single append so lines
# stay intact.
_audit_file_lock = threading.Lock()


def _audit_record(
    request_id: str, payload_hash: str, result: dict[str, Any]
) -> dict[str, Any]:
    """Build one JSONL audit record."""
    return {
        "request_id": request_id,
        "input_hash": payload_hash,
        "fraud_probability": result["fraud_probability"],
        "is_fraud": result["is_fraud"],
        "latency_ms": result["latency_ms"],
        "model_version": result["model_version"],
    }


def _write_audit_records(records: list[dict[str, Any]]) -> None:
    """Append records to the JSONL audit log in a single open (blocking)."""
    if not records:
        return
    log_path = Path(os.getenv("PREDICTION_LOG_PATH", "predictions.jsonl"))
    payload = "".join(json.dumps(record) + "\n" for record in records)
    with _audit_file_lock, log_path.open("a", encoding="utf-8") as handle:
        handle.write(payload)


async def _audit_log(records: list[dict[str, Any]]) -> None:
    """Persist audit records without blocking the event loop.

    The write is pushed to a worker thread: a synchronous open/write/close on
    the loop stalls every other in-flight connection, and a 1000-row batch used
    to do that once per row.
    """
    await asyncio.to_thread(_write_audit_records, records)


def _hash_payload(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 hash of a feature frame."""
    raw = frame.to_json(orient="records").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _score_frame(
    model: Model, frame: pd.DataFrame, threshold: float
) -> list[dict[str, Any]]:
    """Score a feature frame, returning one result dict per row.

    Raw ``Time``/``Amount`` from the request are transformed with the fitted
    training scaler before scoring so the model sees inputs in the same space
    it was trained on (no train/serve skew).
    """
    ordered = frame[list(FEATURE_COLUMNS)].copy()
    scaler = STATE.get("scaler")
    if scaler is not None:
        ordered[list(SCALED_COLUMNS)] = scaler.transform(ordered[list(SCALED_COLUMNS)])
    probs = [float(p) for p in model.predict(ordered)]
    version = STATE["info"].get("version", "unknown")
    results: list[dict[str, Any]] = []
    for prob in probs:
        results.append(
            {
                "fraud_probability": prob,
                "is_fraud": prob >= threshold,
                "threshold_used": threshold,
                "model_version": version,
            }
        )
    return results


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the bare root to the interactive Swagger docs.

    Without this, hitting ``/`` returns 404 (the app only defines the scoring
    and ops endpoints); a landing redirect makes the deployed demo open on the
    interactive API docs instead of a "Not Found".
    """
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness probe reporting loaded-model identity."""
    info = STATE["info"]
    status: Literal["healthy", "degraded"] = "healthy" if _is_ready() else "degraded"
    return HealthResponse(
        status=status,
        model_name=info.get("name", "unknown"),
        model_version=info.get("version", "unknown"),
        model_stage=info.get("stage", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionFeatures,
    request: Request,
    model: Model = Depends(get_model),
) -> PredictionResponse:
    """Score a single transaction."""
    start = time.perf_counter()
    frame = pd.DataFrame([transaction.model_dump()])
    result = _score_frame(model, frame, STATE["threshold"])[0]
    result["latency_ms"] = (time.perf_counter() - start) * 1000.0

    await _audit_log(
        [_audit_record(request.state.request_id, _hash_payload(frame), result)]
    )
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    model: Model = Depends(get_model),
) -> BatchPredictionResponse:
    """Score up to 1000 transactions in one request."""
    start = time.perf_counter()
    frame = pd.DataFrame([t.model_dump() for t in payload.transactions])
    results = _score_frame(model, frame, STATE["threshold"])
    elapsed = (time.perf_counter() - start) * 1000.0

    request_id = request.state.request_id
    # The batch is scored in ONE vectorised call, so there is no per-row timing
    # to report. Each row therefore carries the whole-request latency, and the
    # envelope surfaces it once. (This previously reported `elapsed / n` as if
    # it were a measured per-row latency, which no prediction ever experienced.)
    # Hashing is still per row so each audit line identifies its own input; at
    # the 1000-row cap that cost is bounded and no longer dominated by I/O.
    records: list[dict[str, Any]] = []
    responses: list[PredictionResponse] = []
    for i, result in enumerate(results):
        result["latency_ms"] = elapsed
        records.append(
            _audit_record(f"{request_id}:{i}", _hash_payload(frame.iloc[[i]]), result)
        )
        responses.append(PredictionResponse(**result))
    await _audit_log(records)
    return BatchPredictionResponse(
        predictions=responses, count=len(responses), latency_ms=elapsed
    )
