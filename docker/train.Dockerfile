# syntax=docker/dockerfile:1
#
# Training image: installs the [train] extras of both packages
# (torch, mlflow, optuna, transformers, rich, matplotlib)

# `data/` and `mlflow/` are NOT copied into the image - they are mounted as
# volumes by docker-compose, keeping the image reusable and avoiding baking
# the dataset into a distributable image.

FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/dlkit /app/src/dlkit
COPY src/reviews_semantics /app/src/reviews_semantics

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "/app/src/dlkit[all]" \
        "/app/src/reviews_semantics[train]"

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db

CMD ["python", "-m", "reviews_semantics.cli.main", "train-baseline", \
     "--data-dir", "/app/data", "--output-dir", "/app/models/bilstm-baseline"]
