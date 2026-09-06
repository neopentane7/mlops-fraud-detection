"""Container integration test for the fraud API.

Run against a *live* API (e.g. one started by docker-compose in CD). It polls
``/health`` until the model reports healthy, then sends the sample transaction
to ``/predict`` and asserts a well-formed response. Exits non-zero on failure
so the CD pipeline can gate the Staging -> Production promotion on it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib import error, request

BASE_URL = "http://localhost:8000"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "tests" / "sample_transaction.json"
HEALTH_TIMEOUT_S = 60
POLL_INTERVAL_S = 3


def _get(url: str) -> dict[str, object]:
    with request.urlopen(url, timeout=5) as resp:  # noqa: S310 - fixed localhost URL
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as resp:  # noqa: S310 - fixed localhost URL
        return json.loads(resp.read().decode("utf-8"))


def wait_for_healthy() -> None:
    """Poll /health until the model is loaded or the timeout elapses."""
    deadline = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        try:
            body = _get(f"{BASE_URL}/health")
            if body.get("status") == "healthy":
                print(f"[integration] healthy: {body}")
                return
        except error.URLError as exc:
            print(f"[integration] waiting for API... ({exc})")
        time.sleep(POLL_INTERVAL_S)
    raise SystemExit("[integration] API never became healthy")


def check_prediction() -> None:
    """Send the sample transaction and validate the response contract."""
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    body = _post(f"{BASE_URL}/predict", payload)
    print(f"[integration] /predict -> {body}")

    # Explicit raises, not `assert`: this gates a CD promotion, and asserts are
    # stripped under `python -O`, which would turn the whole check into a no-op.
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"[integration] FAILED: {message} (got {body!r})")

    prob = body.get("fraud_probability")
    _require(
        isinstance(prob, int | float) and not isinstance(prob, bool),
        "fraud_probability is not numeric",
    )
    _require(0.0 <= float(prob) <= 1.0, "fraud_probability outside [0, 1]")
    _require(isinstance(body.get("is_fraud"), bool), "is_fraud missing or not a bool")
    _require("model_version" in body, "model_version missing")


def main() -> int:
    """Run the integration checks."""
    wait_for_healthy()
    check_prediction()
    print("[integration] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
