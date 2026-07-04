# Credit-Card Fraud Detection — End-to-End MLOps Pipeline

[![CI](https://github.com/neopentane7/mlops-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/neopentane7/mlops-fraud-detection/actions/workflows/ci.yml)
[![Security](https://github.com/neopentane7/mlops-fraud-detection/actions/workflows/security.yml/badge.svg)](https://github.com/neopentane7/mlops-fraud-detection/actions/workflows/security.yml)
[![Live demo](https://img.shields.io/badge/live_demo-Hugging_Face_Spaces-ffab00?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/neopentane7/fraud-detection-demo)
[![Container image](https://img.shields.io/badge/image-GHCR-2496ED?logo=docker&logoColor=white)](https://github.com/neopentane7/mlops-fraud-detection/pkgs/container/mlops-fraud-detection%2Ffraud-api)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **production-grade, reproducible, and monitored** machine-learning system for
credit-card fraud detection. Every pipeline stage ships a **programmatic quality
gate that can fail the build** — this is deliberately *not* "just train a model
in a notebook".

**Highlights**

- **Full MLOps loop:** DVC data pipeline → XGBoost with MLflow tracking &
  registry → gated CI/CD → FastAPI serving → Evidently drift monitoring →
  automated retraining on drift.
- **Dataset-agnostic:** the *same* config-driven pipeline runs on **three real
  datasets** — card fraud, credit default, and Bitcoin AML.
- **Shipped, not just coded:** a [live public demo](https://huggingface.co/spaces/neopentane7/fraud-detection-demo)
  serving the real model, a container image on GHCR, and a manual-approval
  production release gate.
- **Honest evaluation:** recall-first thresholding, a temporal-split analysis,
  and a documented concept-drift failure case — no cherry-picked numbers.

> **🔴 Live demo:** score a transaction in your browser at
> **[huggingface.co/spaces/neopentane7/fraud-detection-demo](https://huggingface.co/spaces/neopentane7/fraud-detection-demo)**
> — open `/docs` for the Swagger UI. The Space trains the real model at build
> time (public OpenML mirror, no secrets) and serves the FastAPI app.

```
data source ─▶ DVC download ─▶ Pandera validate ─▶ preprocess ─▶ XGBoost train (MLflow)
            ─▶ evaluate (benchmark + champion/challenger gates) ─▶ MLflow Registry
            ─▶ FastAPI (Docker) ─▶ Evidently drift monitor ─▶ auto-retrain on drift
```

![Threshold trade-off](docs/images/threshold_tradeoff.png)

## Contents

- [Why this is more than a notebook](#why-this-is-more-than-a-notebook)
- [Datasets used for training](#datasets-used-for-training--organic-not-synthetic)
- [Results (three datasets)](#results--three-datasets-comparable-format-public)
- [Quick start](#quick-start-windows--powershell)
- [Repository layout](#repository-layout)
- [Multi-dataset design](#multi-dataset-config-driven)
- [Engineering decisions](#engineering-decisions)
- [CI/CD](#cicd-github-actions)
- [Testing](#testing)
- [Limitations & future work](#limitations--future-work)
- [Documentation](#documentation)

## Documentation

| Document | What's inside |
| --- | --- |
| **[results.md](docs/results.md)** | Consolidated results on all three datasets: metrics, gates, cross-dataset comparison charts, confusion matrices, literature comparison. |
| **[model_card.md](docs/model_card.md)** | Model card — intended use, training data & provenance, metrics, ethical considerations, caveats. |
| **[elliptic_analysis.md](docs/elliptic_analysis.md)** | Elliptic temporal evaluation reproducing Weber et al. (2019): random vs temporal split and the dark-market-shutdown concept-drift collapse. |
| **[architecture.md](docs/architecture.md)** | Mermaid diagrams — system architecture, DVC DAG, MLflow promotion lifecycle, drift→retrain loop, request flow, CI/CD topology. |
| **[analysis.md](docs/analysis.md)** | Charts from the real data — imbalance, PR/ROC, the threshold trade-off, feature importance, `scale_pos_weight` sweep, single-split variance. |
| **[deploy.md](docs/deploy.md)** | Deploy the self-contained live demo (public Swagger + `/predict`) to Render / Hugging Face Spaces, no secrets. |
| **[scaling.md](docs/scaling.md)** | Production & scaling design notes — feature store, streaming, canary deploys, delayed labels, compliance. |

## Why this is more than a notebook

| Stage | Quality gate that can fail the pipeline |
| --- | --- |
| `download` | Row-count plausibility check (≥ 200k rows); Kaggle → OpenML fallback |
| `validate` | Pandera schema: dtypes, ranges, finite PCA columns, plausible fraud rate |
| `preprocess` | Leakage guard — scaler fit on **train only**, dedup before split, stratified |
| `train` | Performance gate — promote to Staging only if `f1_fraud ≥ performance_threshold` |
| `evaluate` | Benchmark gate **and** challenger-must-not-regress-vs-Production gate |
| serving | HTTP 503 (not 500) until a model is actually loaded; scaler applied to match training |
| monitoring | Drift-share threshold (0.30) auto-triggers retraining |

## Datasets used for training — organic, not synthetic

Every model in this repo is trained on a **real, public** dataset (no synthetic
training data). The pipeline ships **three** organic profiles, each fetched by
[src/data/download.py](src/data/download.py) from the source below. The only
synthetic data anywhere is a tiny fixture used by the test suite and the live
demo for hermetic speed — **no reported result is trained on synthetic data.**

| Profile (`MLOPS_DATASET`) | Trained on | Source (exact) | Rows / positives | Imbalance |
| --- | --- | --- | --- | --- |
| `creditcard` *(default)* | European-cardholder card transactions, Sept 2013; features `Time`, `Amount`, PCA `V1–V28` | Kaggle `mlg-ulb/creditcardfraud`, else public OpenML mirror `data_id=42175` (no login) | 284,807 → 283,726 dedup / 492 fraud | **578 : 1** |
| `cc-default` | Taiwan credit-card client default (Yeh & Lien 2009); 23 features `x1–x23` | OpenML `data_id=42477` (no login) | 30,000 → 29,965 / 6,636 | **3.5 : 1** |
| `elliptic` | Real Bitcoin transactions labelled illicit/licit (Weber et al. 2019); 165 node features `f1–f165` | Kaggle `ellipticco/elliptic-data-set` | 46,564 labelled / 4,545 illicit | **9.2 : 1** |

Kaggle sources fall back to a no-login public mirror where one exists, so the
pipeline reproduces anywhere. Full provenance, licences and preprocessing are in
the **[model card](docs/model_card.md)**.

## Results — three datasets, comparable format (public)

Headline metrics on the held-out test split (stratified, for cross-dataset
comparability), produced by the **same** config-driven pipeline. Full tables,
per-dataset gates, figures, and the literature comparison are in
**[docs/results.md](docs/results.md)**; each model is documented in the
**[model card](docs/model_card.md)**.

| Metric (positive class) | `creditcard` (fraud) | `cc-default` (default) | `elliptic` (AML) |
| --- | --- | --- | --- |
| ROC-AUC | **0.971** | **0.772** | **0.995** |
| Average precision (AUPRC) | **0.834** | **0.552** | **0.980** |
| Recall | 0.831 | 0.642 | 0.689 |
| Precision | 0.678 | 0.434 | 0.998 |
| F1 | 0.747 | 0.518 | 0.815 |
| Test positives (support) | 71 | 995 | 682 |
| Own benchmark gate | ✅ passed | ✅ passed | ✅ passed |

![Model performance across all three datasets](docs/images/comparison/metrics_bar.png)

The [full results](docs/results.md) include ROC/PR overlays across all datasets
and the per-dataset figure sets (imbalance, PR/ROC, threshold trade-off,
confusion matrix, feature importance).

> `elliptic` is graph data scored **tabularly on a random split** — optimistic
> for this dataset. On the paper's *temporal* protocol F1 is **0.753** (vs the
> paper's RF 0.787), and a dark-market-shutdown concept-drift collapse appears;
> see [docs/elliptic_analysis.md](docs/elliptic_analysis.md). All three land on
> their published ROC-AUC/AUPRC ceilings — see [docs/results.md](docs/results.md).

## Quick start (Windows / PowerShell)

```powershell
# 1. Environment (Python 3.11) — editable install of the package + dev extras
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"   # extras: train | api | dev (= train+api+tooling)

# 2. (Optional) Kaggle credentials — skip to use the OpenML mirror automatically
Copy-Item .env.example .env   # then fill in KAGGLE_USERNAME / KAGGLE_KEY

# 3. Start the MLflow tracking server (SQLite backend + local artifacts)
docker compose up -d mlflow

# 4. Run the full pipeline (download -> validate -> preprocess -> train -> evaluate)
dvc init --no-scm
dvc repro

# 5. Serve the production model and smoke-test it
docker compose up --build api
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@tests/sample_transaction.json"

# 6. Drift monitoring (dry-run with injected drift)
python src/monitoring/detect_drift.py --simulate-drift
```

`make` shortcuts (`setup`, `pipeline`, `train`, `serve`, `test`, `lint`,
`drift`, `clean`) are in the [Makefile](Makefile).

## Repository layout

```
src/config.py            Typed dataclass config loader + canonical paths (single source of params)
src/data/                download.py (Kaggle/OpenML) · validate.py (Pandera) · preprocess.py
src/models/              train.py (MLflow) · evaluate.py (gates) · predict.py (pyfunc wrapper)
src/monitoring/          detect_drift.py (Evidently)
api/                     main.py (FastAPI lifespan) · schemas.py · Dockerfile
scripts/                 promote_model.py · integration_test.py · pre-commit guard
.github/workflows/       7 workflows: ci · quality · e2e · cd · retrain · monitor · security
dvc.yaml / params.yaml   5-stage pipeline + all tunable parameters
tests/                   hermetic pytest suite (synthetic fixture, 82% coverage)
```

## Multi-dataset (config-driven)

The pipeline is **dataset-agnostic**: the schema (feature/scaled/target columns,
validation bounds), the benchmark gate, and per-dataset training overrides live
as profiles in the `data` section of [params.yaml](params.yaml). Select one with
the `MLOPS_DATASET` env var; outputs are namespaced per dataset so runs never
collide. It ships with three organic profiles — `creditcard` (card fraud),
`cc-default` (credit-card default, OpenML 42477), and `elliptic` (real Bitcoin
transactions labelled illicit/licit, Kaggle `ellipticco/elliptic-data-set`) —
and the *identical* stage code runs on all of them:

```bash
MLOPS_DATASET=cc-default MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  python src/data/download.py && python src/data/validate.py && \
  python src/data/preprocess.py && python src/models/train.py && \
  python src/models/evaluate.py --stage holdout
```

See [docs/second_dataset_demo.md](docs/second_dataset_demo.md) for the full
second-dataset run (it matched the published ROC-AUC ≈ 0.77 ceiling).

## Model performance — measured on the real data

Targets are **calibrated from 5-seed / 5-fold experiments on the real,
deduplicated dataset**, not copied from an aspirational spec. `evaluate` fails
explicitly, naming the offending metric, if any target is missed. Overall
accuracy is never reported — it is meaningless at 577:1.

| Metric | Gate | Measured (holdout) | Why this floor |
| --- | --- | --- | --- |
| `roc_auc` | ≥ 0.96 | ~0.97 | threshold-independent, stable (0.976 ± 0.01 CV) |
| `avg_precision` (AUPRC) | ≥ 0.80 | ~0.83 | the right summary under imbalance (0.82 ± 0.02 CV) |
| `recall_fraud` | ≥ 0.78 | ~0.83 | business priority — catch the fraud |
| `precision_fraud` | ≥ 0.60 | ~0.68 | "not-collapsed" floor; precision-at-fixed-recall is noisy with ~71 holdout frauds |

### Empirical findings

* **Generalization is strong on the stable metrics**: ROC-AUC 0.976 ± 0.01 and
  AUPRC 0.82 ± 0.02 across folds — consistent with published results.
* **The original "precision ≥ 0.82 *and* recall ≥ 0.78 simultaneously" goal is
  not reliably achievable on a single split.** With only ~71–95 frauds per
  evaluation fold and a steep PR curve, the operating point is high-variance
  (precision swung 0.77–0.98, recall 0.68–0.86 across seeds). This is intrinsic
  to the data's extreme rarity — not a code defect.
* **What changed because of that**: recall-first thresholding, exact-duplicate
  removal before splitting (~1k dupes that otherwise leak train→test), and
  CV-tuned hyperparameters (depth-4 / 400 trees / `scale_pos_weight=24`), plus
  benchmark targets recalibrated to what the data supports.

## Engineering decisions

**Why XGBoost over a neural network?** Tabular data, 30 features, sparse
positives — gradient boosting achieves competitive AUC with far less compute
and full SHAP interpretability. A NN would need heavy regularisation to avoid
memorising the rare class.

**Why `scale_pos_weight` over SMOTE?** SMOTE fabricates synthetic frauds by
interpolation that may not represent real attack vectors; reweighting the loss
adjusts for imbalance without inventing data. The value was **tuned to 24** —
the naive 577 destroyed probability calibration without improving AUPRC.

**Why recall-first thresholding (not max-F1, not 0.5)?** Maximising F1 drifts to
a precision-heavy threshold (≈0.99) that drops recall to ~0.70 — contradicting
the business priority. The pipeline picks the highest-precision threshold whose
validation recall ≥ `min_recall` (0.85): the PR-curve boundary. A missed fraud
costs far more than a false alarm.

**Why the MLflow Model Registry over model files?** Deployment is a registry
stage transition (`Staging → Production`) — no code change, no redeploy, fully
auditable. The API always loads `models:/fraud-detector/Production`.

**Why apply the scaler at serving time?** The model trains on scaled
`Time`/`Amount`; the API receives raw values. The fitted `RobustScaler` is
loaded at startup and applied before scoring to eliminate train/serve skew.

**Why Evidently over hand-rolled tests?** It picks the right statistical test
per column and emits both HTML (humans) and JSON (the decision gate). The 0.30
drift-share threshold is conservative — over-triggering retraining is cheaper
than missing real distribution shift in fraud.

## Edge cases found and fixed

| Severity | Issue | Fix |
| --- | --- | --- |
| 🔴 Critical | Train/serve skew — API fed **raw** `Time`/`Amount` to a model trained on **scaled** values | Load + apply the fitted scaler at serving ([api/main.py](api/main.py)) |
| 🟠 High | `NaN`/`Infinity` accepted by JSON floats, corrupting scoring | Pydantic `field_validator` rejects non-finite ([api/schemas.py](api/schemas.py)) |
| 🟠 High | Degenerate threshold tuning on zero-fraud / flat-probability splits | Recall-floor with F1/0.5 fallbacks ([src/models/train.py](src/models/train.py)) |
| 🟠 High | Max-F1 threshold sacrificed recall (business inversion) | Recall-first thresholding |
| 🟡 Medium | ~1,081 duplicate rows leaking train→test | `drop_duplicates()` before split ([src/data/preprocess.py](src/data/preprocess.py)) |
| 🟡 Medium | `load_threshold()` default-arg bound at import (config ignored) | Resolve module global at call time ([src/models/evaluate.py](src/models/evaluate.py)) |

## CI/CD (GitHub Actions)

Seven workflows, each with a distinct responsibility. All run without secrets
unless noted.

* **ci.yml** (push/PR) — ruff, mypy, pytest (≥ 80% coverage), and DVC DAG
  validation. With Kaggle secrets configured, it additionally runs a quick train,
  an `f1_fraud` gate, and a metrics pull-request comment.
* **quality.yml** (push/PR) — downloads `cc-default` from OpenML and enforces its
  benchmark gate (roc_auc ≥ 0.74, recall ≥ 0.55, …); a regression below the bar
  fails the build. Real-data quality assurance without secrets.
* **e2e.yml** (push/PR) — the full loop on the synthetic profile: validate →
  preprocess → train → register + promote in a local SQLite registry → serve via
  FastAPI → `POST /predict`. Driver: [scripts/e2e_smoke.py](scripts/e2e_smoke.py).
* **cd.yml** (manual) — Continuous Delivery. Stage 1 (automated): train → holdout
  gate → build image → Trivy image scan (blocks on fixable CRITICALs) → container
  integration test → publish a SHA-versioned candidate to GHCR. Stage 2
  (`release-production`): promote the candidate to `:latest`, gated by a manual
  approval via the `production` GitHub Environment (configure "Required
  reviewers" in repository settings).
* **retrain.yml** (manual / drift dispatch) — retrains a challenger, evaluates it,
  and promotes only when it beats the Production model's `f1_fraud` by ≥ 2%.
* **monitor.yml** (weekly cron / manual) — builds a reference baseline and an
  Evidently drift report; the cron performs real detection and dispatches
  `retrain` only when the drift share exceeds the threshold.
* **security.yml** (push/PR/weekly) — pip-audit, CodeQL, and Trivy. Dependency and
  filesystem scans report without blocking; CodeQL results upload to the Security
  tab.

The loop workflows (`e2e`, `retrain`, `monitor`, `cd`) validate pipeline
mechanics on the synthetic profile; `quality` validates model quality on real
data. Configuring the real datasets and a hosted MLflow server via secrets runs
the identical code on production infrastructure.

**Reproducible installs.** CI and the serving image install from fully-pinned
lockfiles (`requirements.lock`, `requirements-api.lock`) so transitive versions
can't drift between runs. Regenerate them after changing `pyproject.toml`:

```powershell
uv pip compile --python-platform linux --python-version 3.11 --extra dev -o requirements.lock pyproject.toml
uv pip compile --python-platform linux --python-version 3.11 --extra api -o requirements-api.lock pyproject.toml
```

## Testing

A hermetic test suite with an 80% coverage gate, ruff-clean. It uses a synthetic
fixture, so it requires no Kaggle credentials, network, or tracking server, while
still exercising the full code paths (validation, preprocessing, XGBoost
training, MLflow logging, the pyfunc wrapper, the async API, and drift logic).

```powershell
pytest tests/ -v --cov=src --cov-fail-under=80
ruff check src/ tests/ api/
```

## Modelling choices

* **Split strategy** (`preprocess.split_strategy`): default `stratified` (random,
  comparable to the published random-split benchmarks). An opt-in `temporal` mode
  trains on the oldest transactions and tests on the newest — a stricter,
  no-leakage evaluation that better reflects production; on `creditcard` it is
  ~0.07 lower AUPRC — the cost of evaluating on future data. Falls back to
  stratified when the profile has no time column (cc-default).
* **Decision threshold** is tuned on validation, not left at 0.5. Two strategies
  (`train.threshold_strategy`): `recall` (highest precision above a recall floor)
  or `cost` (minimise `cost_fn*FN + cost_fp*FP` — fraud's asymmetric error costs).
* **Probability calibration** (`train.calibration: isotonic`): an isotonic map
  fit on validation corrects the scores `scale_pos_weight` distorts, bundled into
  the serving model so `/predict` returns a trustworthy probability. Rank-
  preserving, so ROC-AUC / AUPRC are unchanged.

## Limitations & future work

* **Single-split benchmark gating is noisy** at this fraud count — the most
  robust next step is **K-fold CV-based gating** (gate on the mean across folds).
* Features arrive pre-PCA'd, which limits how narratable SHAP explanations are.
* No online feature store — serving recomputes from the request payload only.

## Stack

Python 3.11 · DVC · Pandera · XGBoost · scikit-learn · MLflow · SHAP ·
Evidently · FastAPI · Pydantic v2 · Docker · GitHub Actions · pytest · ruff · mypy

## License

Released under the [MIT License](LICENSE).
