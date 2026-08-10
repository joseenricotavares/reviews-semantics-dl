# syntax=docker/dockerfile:1
#
# Lean production image for the online prediction API + offline batch CLI.
# Installs only `dlkit` (no extras) and `reviews_semantics` (no [train]
# extra) - no mlflow/optuna/transformers/matplotlib/seaborn/rich here, only
# what's needed to load the committed BiLSTM bundle and serve predictions.

FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip

COPY src/dlkit /build/src/dlkit
COPY src/reviews_semantics /build/src/reviews_semantics

RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    /build/src/dlkit /build/src/reviews_semantics


FROM python:3.12-slim AS runtime

# torch needs libgomp1 (OpenMP) at runtime, which python:3.12-slim omits by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY models/bilstm-baseline /app/model

ENV MODEL_SOURCE=local \
    MODEL_PATH=/app/model \
    DEVICE=cpu \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()" || exit 1

CMD ["uvicorn", "reviews_semantics.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
