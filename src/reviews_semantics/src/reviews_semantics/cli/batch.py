from __future__ import annotations

import csv
from pathlib import Path

from dlkit.inference import Predictor

from reviews_semantics.inference import load_default_predictor

DEFAULT_CHUNK_SIZE = 256


def predict_csv(
    input_path: Path,
    output_path: Path,
    *,
    text_column: str = "review_comment_message",
    score_column: str = "predicted_score",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    predictor: Predictor | None = None,
) -> int:
    """Reads `input_path`, predicts a star rating for each row's `text_column`,
    and writes `output_path` with an added `score_column`. Returns the number
    of rows processed.

    Streams in fixed-size chunks so arbitrarily large CSVs don't need to fit
    in memory at once.
    """
    predictor = predictor or load_default_predictor()

    with input_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(f"{input_path} has no column named {text_column!r}")
        fieldnames = [*reader.fieldnames, score_column]

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            rows_processed = 0
            chunk: list[dict[str, str]] = []
            for row in reader:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    rows_processed += _flush_chunk(chunk, predictor, text_column, score_column, writer)
                    chunk = []
            if chunk:
                rows_processed += _flush_chunk(chunk, predictor, text_column, score_column, writer)

    return rows_processed


def _flush_chunk(
    chunk: list[dict[str, str]],
    predictor: Predictor,
    text_column: str,
    score_column: str,
    writer: csv.DictWriter,
) -> int:
    scores = predictor.predict([row[text_column] for row in chunk])
    for row, score in zip(chunk, scores, strict=True):
        row[score_column] = str(score)
        writer.writerow(row)
    return len(chunk)
