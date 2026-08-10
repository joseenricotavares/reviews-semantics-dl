from __future__ import annotations

from collections.abc import Sequence

import torch
from dlkit.artifacts import ArtifactBundle
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reviews_semantics.labels import STAR_SCHEMA

TR_MAX_LEN = 128


class TransformerPredictor:
    """Raw text -> predicted star rating using a fine-tuned HuggingFace model."""

    label_schema = STAR_SCHEMA

    def __init__(self, model, tokenizer, device: str, max_len: int = TR_MAX_LEN) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    @classmethod
    def load(cls, bundle: ArtifactBundle, device: str = "cpu") -> TransformerPredictor:
        checkpoint_dir = bundle.path_for("hf_checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=device)

    @torch.no_grad()
    def predict(self, texts: Sequence[str]) -> list[int]:
        encodings = self.tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        logits = self.model(**encodings).logits
        return (torch.argmax(logits, dim=1) + 1).cpu().tolist()
