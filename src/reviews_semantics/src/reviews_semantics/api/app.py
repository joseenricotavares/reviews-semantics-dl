from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from reviews_semantics.api.routers import health, predict
from reviews_semantics.inference import load_default_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    loaded = load_default_model()
    app.state.predictor = loaded.predictor
    app.state.bundle = loaded.bundle
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Reviews Semantics API",
        description=(
            "Predicts the 1-5 star rating of a Brazilian e-commerce review "
            "(Olist dataset) from its free-text comment, using a BiLSTM baseline."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/v1")
    return app


app = create_app()
