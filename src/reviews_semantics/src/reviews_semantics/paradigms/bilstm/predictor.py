from __future__ import annotations

import json
from collections.abc import Sequence

import torch
from dlkit.artifacts import ArtifactBundle

from reviews_semantics.data.cleaning import clean_text
from reviews_semantics.data.vocab import TextPreprocessor
from reviews_semantics.labels import STAR_SCHEMA
from reviews_semantics.paradigms.bilstm.model import SentimentLSTM


class BiLSTMPredictor:
    """Raw text -> predicted star rating, using the trained BiLSTM baseline."""

    label_schema = STAR_SCHEMA

    def __init__(self, model: SentimentLSTM, preprocessor: TextPreprocessor, device: str) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.device = device

    @classmethod
    def load(cls, bundle: ArtifactBundle, device: str = "cpu") -> BiLSTMPredictor:
        config = json.loads(bundle.path_for("model_config").read_text(encoding="utf-8"))
        model = SentimentLSTM(**config).to(device)
        model.load_state_dict(torch.load(bundle.path_for("weights"), map_location=device))
        model.eval()

        preprocessor = TextPreprocessor.from_dict(
            json.loads(bundle.path_for("preprocessor").read_text(encoding="utf-8"))
        )
        return cls(model=model, preprocessor=preprocessor, device=device)

    @torch.no_grad()
    def predict(self, texts: Sequence[str]) -> list[int]:
        cleaned = [clean_text(text) for text in texts]
        sequences = [self.preprocessor.encode(text) for text in cleaned]
        batch = torch.tensor(sequences, dtype=torch.long, device=self.device)
        logits = self.model(batch)
        return (torch.argmax(logits, dim=1) + 1).cpu().tolist()
