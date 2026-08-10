from __future__ import annotations

import json
from collections.abc import Sequence

import joblib
from dlkit.artifacts import ArtifactBundle
from transformers import AutoTokenizer

from reviews_semantics.labels import STAR_SCHEMA
from reviews_semantics.paradigms.feature_extraction.trainer import TransformerFeatureExtractor


class FeatureExtractionPredictor:
    """Raw text -> predicted star rating using a frozen encoder + LogisticRegression head."""

    label_schema = STAR_SCHEMA

    def __init__(self, extractor: TransformerFeatureExtractor) -> None:
        self.extractor = extractor

    @classmethod
    def load(cls, bundle: ArtifactBundle, device: str = "cpu") -> FeatureExtractionPredictor:
        config = json.loads(bundle.path_for("config").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
        extractor = TransformerFeatureExtractor(
            config["model_path"],
            tokenizer,
            device,
            model_id=config.get("model_id"),
            max_len=config["max_len"],
        )
        extractor.classifier = joblib.load(bundle.path_for("classifier"))
        return cls(extractor)

    def predict(self, texts: Sequence[str]) -> list[int]:
        return self.extractor.predict(texts)
