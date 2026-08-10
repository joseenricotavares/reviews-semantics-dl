from __future__ import annotations

import gc
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import torch
from dlkit.artifacts import ArtifactBundle, BundleFile
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoModel

from reviews_semantics.data.dataset import TransformerDataset
from reviews_semantics.labels import STAR_SCHEMA

TR_MAX_LEN = 128
TR_BATCH_SIZE = 32


class TransformerFeatureExtractor:
    """Freezes a pre-trained encoder, extracts embeddings, and fits a
    Logistic Regression classifier on top."""

    def __init__(self, model_path: str, tokenizer, device: torch.device, model_id: str | None = None,
                 max_len: int = TR_MAX_LEN, batch_size: int = TR_BATCH_SIZE) -> None:
        self.model_path = model_path
        self.model_id = model_id or model_path
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.classifier = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        )

    @torch.no_grad()
    def _embed(self, X: Sequence[str]) -> np.ndarray:
        """Loads the frozen encoder, extracts the [CLS] embedding per example,
        then releases the encoder before returning."""
        encoder = AutoModel.from_pretrained(self.model_path).to(self.device).eval()
        loader = DataLoader(
            TransformerDataset(X, [0] * len(X), self.tokenizer, self.max_len),
            batch_size=self.batch_size,
            shuffle=False,
        )
        embeddings = []
        try:
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        finally:
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
        return np.vstack(embeddings)

    def fit(self, X_train: Sequence[str], y_train: Sequence[int]) -> TransformerFeatureExtractor:
        self.classifier.fit(self._embed(X_train), y_train)
        return self

    def predict(self, X: Sequence[str]) -> list[int]:
        return (self.classifier.predict(self._embed(X)) + 1).tolist()

    def _write_config(self, path: Path) -> None:
        config = {"model_path": str(self.model_path), "model_id": self.model_id, "max_len": self.max_len}
        path.write_text(json.dumps(config), encoding="utf-8")

    def save(self, output_dir: str | Path) -> ArtifactBundle:
        # The encoder is not re-saved: it already lives at self.model_path.
        return ArtifactBundle.write(
            output_dir,
            flavor="feature_extraction",
            label_schema=STAR_SCHEMA,
            files={
                "classifier": BundleFile("classifier.joblib", lambda p: joblib.dump(self.classifier, p)),
                "config": BundleFile("config.json", self._write_config),
            },
        )
