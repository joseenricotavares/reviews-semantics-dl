from __future__ import annotations

from dlkit.artifacts import LoadedModel
from dlkit.inference import Predictor

from reviews_semantics.config import Settings
from reviews_semantics.registry import build_registry_from_env


def load_default_model(settings: Settings | None = None) -> LoadedModel:
    """Loads the predictor plus the artifact bundle it came from (needed for
    the `/v1/model/info` endpoint's manifest)."""
    return build_registry_from_env(settings).load_with_bundle()


def load_default_predictor(settings: Settings | None = None) -> Predictor:
    """The single entrypoint used by both the API and the batch CLI to load
    the model to serve - both call this (or `load_default_model`) and
    nothing else, so there is exactly one code path for "how do we get a
    ready-to-use predictor"."""
    return build_registry_from_env(settings).load()
