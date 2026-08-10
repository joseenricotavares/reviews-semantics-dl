from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from reviews_semantics.data.vocab import TextPreprocessor


class ReviewDataset(Dataset):
    """Wraps already-cleaned review texts + star labels for the BiLSTM paradigm."""

    def __init__(self, X: Sequence[str], y: Sequence[int], preprocessor: TextPreprocessor) -> None:
        self.X = [preprocessor.encode(text) for text in X]
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


class TransformerDataset(Dataset):
    """Tokenizes raw review text into (input_ids, attention_mask, labels) samples."""

    def __init__(self, X: Iterable[str], y: Sequence[int], tokenizer, max_len: int) -> None:
        self.encodings = tokenizer(
            list(X), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        self.y = list(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.y[idx], dtype=torch.long),
        }
