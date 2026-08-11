from __future__ import annotations

from dlkit.artifacts import (
    LoadedModel,
    LocalPathSource,
    MLflowRunSource,
    ModelRegistry,
    ModelSource,
    PredictorFactory,
)
from dlkit.inference import Predictor

from reviews_semantics.paradigms.bilstm.predictor import BiLSTMPredictor
from reviews_semantics.serving_config import Settings

# Paradigms 2 (transformer_finetune) and 3 (feature_extraction) are
# deliberately absent here: no trained artifact for either is committed to
# this repository. Their Predictor stubs still satisfy dlkit.inference.Predictor
# (see tests/unit/test_stub_predictors_protocol.py) - wiring one of them into
# serving is a one-line addition here once a real artifact exists; nothing
# else (API, CLI, Docker image) has to change.
DEFAULT_FACTORIES: dict[str, PredictorFactory] = {
    "bilstm": BiLSTMPredictor.load,
}


def build_registry_from_env(settings: Settings | None = None) -> ModelRegistry:
    settings = settings or Settings()

    source: ModelSource
    if settings.model_source == "local":
        source = LocalPathSource(settings.model_path)
    else:
        if not settings.mlflow_run_id:
            raise ValueError("MLFLOW_RUN_ID must be set when MODEL_SOURCE=mlflow")
        source = MLflowRunSource(settings.mlflow_run_id, tracking_uri=settings.mlflow_tracking_uri)

    return ModelRegistry(source=source, factories=DEFAULT_FACTORIES, device=settings.device)


def load_default_model(settings: Settings | None = None) -> LoadedModel:
    """Loads the predictor plus the artifact bundle it came from (needed for
    the `/v1/model/info` endpoint's manifest)."""
    return build_registry_from_env(settings).load_with_bundle()


def load_default_predictor(settings: Settings | None = None) -> Predictor[str]:
    """The single entrypoint used by both the API and the batch CLI to load
    the model to serve - both call this (or `load_default_model`) and
    nothing else, so there is exactly one code path for "how do we get a
    ready-to-use predictor"."""
    return build_registry_from_env(settings).load()
