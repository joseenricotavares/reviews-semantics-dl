from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from dlkit.artifacts.bundle import ArtifactBundle, LabelSchema


@runtime_checkable
class Predictor(Protocol):
    """The inference interface every paradigm implements."""

    label_schema: LabelSchema

    def predict(self, texts: Sequence[str]) -> list[int]:
        """Raw input in, predicted label out - including any cleaning/tokenization."""
        ...

    @classmethod
    def load(cls, bundle: ArtifactBundle, device: str = "cpu") -> Predictor:
        """Reconstructs a ready-to-use predictor from a bundle written by `NNTrainer.save()`."""
        ...


@runtime_checkable
class ProbabilisticPredictor(Predictor, Protocol):
    """Optional extension for predictors that can also expose class probabilities."""

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        ...
