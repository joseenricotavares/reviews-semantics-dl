# reviews-semantics-dl

Predicts the 1-5 star rating a customer gave a Brazilian e-commerce order
from the free-text review comment alone (Portuguese), served through a
FastAPI endpoint and an offline batch CLI backed by the same inference code.

The served baseline (a from-scratch bidirectional LSTM) was chosen out of
seven models compared across three paradigms for being competitive with
fine-tuned Transformers on macro-F1 while being far cheaper to run. Training
and serving were designed together from the start: [`dlkit`](src/dlkit/README.md)
is a reusable PyTorch training toolkit plus a framework-agnostic
artifact/serving contract, extracted along the way; `reviews_semantics`
(`src/reviews_semantics/`) is the concrete application built on top of it.

## What's in this repository

| Path | What it is |
|---|---|
| `notebooks/review-score-semantics.ipynb` | The original research notebook: trains and compares 7 models across 3 paradigms (BiLSTM from scratch, 3 fine-tuned Transformers, 3 frozen-encoder feature-extraction models) |
| `src/dlkit/` | Reusable, domain-agnostic epoch-based PyTorch training toolkit + framework-agnostic model artifact/serving contracts - see [`src/dlkit/README.md`](src/dlkit/README.md) |
| `src/reviews_semantics/` | The Olist review-score application: data prep, the BiLSTM model, the FastAPI service, the batch CLI - built on top of `dlkit` |
| `models/bilstm-baseline/` | The committed, ready-to-serve model bundle |
| `docker/`, `docker-compose.yml` | Container definitions for serving, (re)training, and MLServer |
| `mlserver/` | MLServer configuration (`model-settings.json`) for the alternative serving path |
| `.gitlab-ci.yml` | GitLab CI/CD pipeline (test, MLServer smoke test, deploy to Docker Hub) |

## Quickstart (Docker)

Requires Docker and Docker Compose. From the repository root:

```bash
docker compose up --build serve
```

This builds the lean serving image and starts the API on `http://localhost:8000`.

```bash
curl http://localhost:8000/healthz
# {"status":"ok"}

curl -X POST http://localhost:8000/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{"text": "Produto excelente, chegou antes do prazo, recomendo!"}'
# {"score":5}
```

Interactive API docs (Swagger UI): `http://localhost:8000/docs`.

## The online API

Base URL: `http://localhost:8000` (or wherever the container is deployed).
All endpoints are JSON in, JSON out.

### `POST /v1/predictions`

Predicts the star rating for a single review.

**Request body**

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | Raw review comment, in Portuguese. Non-empty. |

```json
{"text": "Produto excelente, chegou antes do prazo, recomendo!"}
```

**Response body** — `200 OK`

| Field | Type | Notes |
|---|---|---|
| `score` | integer | Predicted star rating, `1`-`5` |

```json
{"score": 5}
```

`422 Unprocessable Entity` if `text` is missing or empty.

### `POST /v1/predictions/batch`

Same as above, for up to 256 texts in one request.

**Request body**

| Field | Type | Required | Notes |
|---|---|---|---|
| `texts` | array of string | yes | 1-256 items |

```json
{"texts": ["Produto ótimo!", "Não gostei, veio errado."]}
```

**Response body** — `200 OK`

| Field | Type | Notes |
|---|---|---|
| `scores` | array of integer | One score per input text, same order |

```json
{"scores": [5, 2]}
```

`422 Unprocessable Entity` if `texts` is empty or has more than 256 items.

### `GET /v1/model/info`

Returns metadata about the currently loaded model - useful to confirm
what's actually being served and to sanity-check its offline metrics.

**Response body** — `200 OK`

```json
{
  "flavor": "bilstm",
  "label_schema_kind": "ordinal",
  "classes": [1, 2, 3, 4, 5],
  "metrics": {
    "test_accuracy": 0.4214,
    "test_macro_f1": 0.4191,
    "test_mae": 0.7887,
    "test_rmse": 1.1454,
    "test_qwk": 0.6793
  }
}
```

### `GET /healthz` / `GET /readyz`

Liveness and readiness probes for orchestrators (Docker healthcheck,
Kubernetes, etc.). `/healthz` always returns `200`; `/readyz` returns `503`
until the model has finished loading at startup.

## The offline batch CLI

Scores every row of a CSV file - the offline-execution case, sharing the
exact same model-loading and inference code as the API (see "Architecture").

```bash
docker compose run --rm serve \
  reviews-semantics predict-batch \
  --input /app/data/test.csv \
  --output /app/out/scored.csv \
  --text-column review_comment_message
```

(Mount your own input/output directories via `-v` if not using
`docker-compose.yml`'s defaults, or run it directly against a local install
- see "Local development" below.)

**What happens**: the input CSV is streamed in chunks of 256 rows (so
arbitrarily large files never need to fit in memory at once); each row's
`--text-column` is scored using the same BiLSTM baseline as the API; the
output CSV is the input CSV plus one new column, `predicted_score`
(configurable via `--score-column`).

**Flags**

| Flag | Default | Meaning |
|---|---|---|
| `--input` | *(required)* | Path to the input CSV |
| `--output` | *(required)* | Path to write the scored CSV |
| `--text-column` | `review_comment_message` | Column containing the review text |

Exits non-zero with a clear error if `--text-column` doesn't exist in the
input file.

## Model serving with MLServer

Besides the FastAPI service above, the same BiLSTM predictor can also be served through
[MLServer](https://mlserver.readthedocs.io/) (an open-source, multi-model inference server
implementing the V2/Open Inference Protocol), via a small custom runtime
(`reviews_semantics/mlserver_runtime.py`) that wraps the same `load_default_predictor()`
used everywhere else in this project.

```bash
docker compose --profile mlserver up --build mlserver
```

Exposes REST on `http://localhost:8080` and gRPC on `:8081`.

```bash
curl -X POST http://localhost:8080/v2/models/review-score-bilstm/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "text", "shape": [1], "datatype": "BYTES", "data": ["Produto otimo, chegou rapido"]}]}'
```

## CI/CD (GitLab)

`.gitlab-ci.yml` at the repository root defines three stages:

| Stage | Job | What it does |
|---|---|---|
| `test` | `test` | Installs both packages, runs `ruff check` and `pytest` - the same checks as the GitHub Actions pipeline. |
| `build` | `mlserver-smoke-test` | Builds the MLServer image and confirms it starts and stays up. |
| `deploy` | `deploy` | Builds the serving image and pushes it to Docker Hub, tagged by commit and as `latest` - runs only on `main`. |

`DOCKERHUB_USER`/`DOCKERHUB_TOKEN` are configured as masked CI/CD variables in the GitLab project settings.

## Architecture

```
                      ┌──────────────────────┐
                      │  dlkit (generic)     │
                      │  - NNTrainer (epochs)│
                      │  - ArtifactBundle    │
                      │  - Predictor protocol│
                      │  - Metric/Evaluator  │
                      └──────────┬───────────┘
                                 │ depends on
                      ┌──────────▼───────────┐
                      │ reviews_semantics    │
                      │  - SentimentLSTM     │
                      │  - BiLSTMPredictor   │
                      └───┬──────────────┬───┘
                          │              │
                 ┌────────▼────┐    ┌────▼────────┐
                 │ FastAPI app │    │ batch CLI   │
                 │ (online)    │    │ (offline)   │
                 └─────────────┘    └─────────────┘
```

The two packages are separately versioned and independently installable
(each has its own `pyproject.toml`); `reviews_semantics` depends on `dlkit`.

**Training and serving share one artifact format.** `dlkit.artifacts.ArtifactBundle`
is the single contract every trainer's `save()` writes and every predictor's
`load()` reads (a `bundle.json` manifest + the model's files). This is what
lets the API and the batch CLI both resolve "which model, from where" through
one `ModelRegistry` (`reviews_semantics/registry.py`), switchable via
environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_SOURCE` | `local` | `local` (a path baked into the image) or `mlflow` (download from an MLflow run - for development) |
| `MODEL_PATH` | `/app/model` | Bundle directory, when `MODEL_SOURCE=local` |
| `MLFLOW_RUN_ID` | *(none)* | MLflow run to download from, when `MODEL_SOURCE=mlflow` |
| `MLFLOW_TRACKING_URI` | *(none)* | Passed to `mlflow.set_tracking_uri` |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

Two other paradigms from the notebook (Transformer fine-tuning, frozen-
encoder feature extraction) have minimal, working implementations under
`src/reviews_semantics/src/reviews_semantics/paradigms/` proving `dlkit`'s
contracts aren't BiLSTM-specific. They are **not** wired into the API/CLI:
no trained artifact for them is committed to this repository.

## Local development (no Docker)

Requires Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e src/dlkit
pip install -e "src/reviews_semantics[train]"   # omit [train] for a lean, serving-only environment
pip install -r requirements-dev.txt              # pytest, ruff, mypy, httpx
```

Run the API:

```bash
MODEL_SOURCE=local MODEL_PATH=models/bilstm-baseline reviews-semantics serve
```

Run the batch CLI:

```bash
MODEL_SOURCE=local MODEL_PATH=models/bilstm-baseline reviews-semantics predict-batch \
  --input data/test.csv --output scored.csv
```

## Using `dlkit` / `reviews_semantics` in another project

Both packages install directly from this repository - no PyPI publish step
required. Point `pip` at a Git URL with `#subdirectory=`:

```bash
pip install "git+https://github.com/joseenricotavares/reviews-semantics-dl.git@dlkit-v0.2.0#subdirectory=src/dlkit"
pip install "git+https://github.com/joseenricotavares/reviews-semantics-dl.git@reviews-semantics-v0.2.0#subdirectory=src/reviews_semantics"
```

Same syntax works with extras (e.g. `"dlkit[mlflow] @ git+...#subdirectory=src/dlkit"`) and in a `requirements.txt`:

```
dlkit @ git+https://github.com/joseenricotavares/reviews-semantics-dl.git@dlkit-v0.2.0#subdirectory=src/dlkit
reviews-semantics @ git+https://github.com/joseenricotavares/reviews-semantics-dl.git@reviews-semantics-v0.2.0#subdirectory=src/reviews_semantics
```

## Reproducing training

The baseline model served in production is a repackaged copy of the
notebook's original checkpoint, not a fresh retrain - this preserves exact
parity with the metrics already reported by the notebook, instead of
depending on a new, non-deterministic training run. The full training
pipeline is nonetheless present and runnable, either in the training
container or locally, for reproducibility and future retraining.

**Docker** (heavier image: torch, MLflow, Optuna, matplotlib):

```bash
docker compose --profile train run --rm train
```

Mounts `./data` (read-only), `./mlflow`, and `./models` as volumes; writes a
freshly trained bundle to `models/bilstm-baseline/` and logs the run to the
local MLflow store. Browse it with:

```bash
docker compose --profile mlflow up mlflow-ui
# http://localhost:5000
```

**Locally**, with the `[train]` extra installed (see above):

```bash
reviews-semantics train-baseline --data-dir data --output-dir models/bilstm-baseline --epochs 20
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest        # runs both packages' test suites (dlkit + reviews_semantics)
ruff check src
```

The `reviews_semantics` suite includes a regression test
(`src/reviews_semantics/tests/regression/test_predictions_parity.py`) that
reproduces the notebook's saved test-set predictions row-for-row through the
migrated `BiLSTMPredictor`, guarding against silent behavior drift from the
migration.

## License

See [`LICENSE`](LICENSE).
