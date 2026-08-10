from __future__ import annotations

from dlkit.artifacts import ArtifactBundle
from dlkit.inference import Predictor
from fastapi import Request


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def get_bundle(request: Request) -> ArtifactBundle:
    return request.app.state.bundle
