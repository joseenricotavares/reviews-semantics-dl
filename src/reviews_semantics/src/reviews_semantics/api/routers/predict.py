from __future__ import annotations

from dlkit.artifacts import ArtifactBundle
from dlkit.inference import Predictor
from fastapi import APIRouter, Depends

from reviews_semantics.api.dependencies import get_bundle, get_predictor
from reviews_semantics.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(tags=["predictions"])


@router.post("/predictions", response_model=PredictionResponse)
def predict_one(
    request: PredictionRequest, predictor: Predictor[str] = Depends(get_predictor)
) -> PredictionResponse:
    (score,) = predictor.predict([request.text])
    return PredictionResponse(score=score)


@router.post("/predictions/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest, predictor: Predictor[str] = Depends(get_predictor)
) -> BatchPredictionResponse:
    return BatchPredictionResponse(scores=predictor.predict(request.texts))


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(bundle: ArtifactBundle = Depends(get_bundle)) -> ModelInfoResponse:
    return ModelInfoResponse(
        flavor=bundle.manifest.flavor,
        label_schema_kind=bundle.manifest.label_schema.kind,
        classes=list(bundle.manifest.label_schema.classes),
        metrics=bundle.manifest.metrics,
    )
