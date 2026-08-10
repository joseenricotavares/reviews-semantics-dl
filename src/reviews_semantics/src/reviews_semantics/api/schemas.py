from __future__ import annotations

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 256


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw review comment text, in Portuguese.")


class PredictionResponse(BaseModel):
    score: int = Field(..., description="Predicted star rating (1-5).")


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE, description="Raw review comment texts."
    )


class BatchPredictionResponse(BaseModel):
    scores: list[int] = Field(..., description="Predicted star rating (1-5) for each input text, same order.")


class ModelInfoResponse(BaseModel):
    flavor: str
    label_schema_kind: str
    classes: list[int]
    metrics: dict[str, float]
