# syntax=docker/dockerfile:1
#
# MLServer image: hosts the same BiLSTM predictor used by the FastAPI service, 
# through MLServer's V2 inference protocol.

FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip

COPY src/dlkit /build/src/dlkit
COPY src/reviews_semantics /build/src/reviews_semantics

RUN pip install --no-cache-dir --prefix=/install mlserver \
    && pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        /build/src/dlkit /build/src/reviews_semantics


FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY models/bilstm-baseline /app/model
COPY mlserver/model-settings.json /app/model-settings.json
COPY mlserver/settings.json /app/settings.json
RUN chown -R appuser:appuser /app

ENV MODEL_SOURCE=local \
    MODEL_PATH=/app/model \
    DEVICE=cpu \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8080 8081

CMD ["mlserver", "start", "/app"]