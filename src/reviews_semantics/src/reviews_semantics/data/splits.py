from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_COLUMNS = ("review_comment_message", "review_score")


def load_split(path: str | Path) -> tuple[list[str], list[int]]:
    """Reads a `review_comment_message,review_score` CSV split (as produced
    by the notebook's train/val/test export) into parallel lists."""
    path = Path(path)
    texts: list[str] = []
    scores: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            texts.append(row["review_comment_message"])
            scores.append(int(row["review_score"]))
    return texts, scores
