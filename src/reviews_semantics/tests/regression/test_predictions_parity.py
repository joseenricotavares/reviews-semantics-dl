from __future__ import annotations

import csv
from pathlib import Path

import pytest
from dlkit.artifacts import ArtifactBundle
from reviews_semantics.data.splits import load_split
from reviews_semantics.paradigms.bilstm.predictor import BiLSTMPredictor

REPO_ROOT = Path(__file__).resolve().parents[4]
BUNDLE_DIR = REPO_ROOT / "models" / "bilstm-baseline"
TEST_SPLIT = REPO_ROOT / "data" / "test.csv"
FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "predictions_Baseline_BiLSTM_test.csv"

pytestmark = pytest.mark.skipif(
    not BUNDLE_DIR.exists() or not TEST_SPLIT.exists(),
    reason="models/bilstm-baseline/ or data/test.csv missing - run scripts/backfill_baseline_bundle.py first",
)


def _load_fixture_predictions() -> list[tuple[int, int]]:
    with FIXTURE.open(newline="", encoding="utf-8") as f:
        return [(int(row["y_true"]), int(row["y_pred"])) for row in csv.DictReader(f)]


def test_bilstm_predictor_matches_notebooks_saved_test_predictions():
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(BUNDLE_DIR))
    texts, true_scores = load_split(TEST_SPLIT)
    expected = _load_fixture_predictions()

    assert len(texts) == len(expected)

    predicted = predictor.predict(texts)

    for i, ((expected_true, expected_pred), actual_true, actual_pred) in enumerate(
        zip(expected, true_scores, predicted, strict=True)
    ):
        assert actual_true == expected_true, f"row {i}: true label drifted ({actual_true} != {expected_true})"
        assert actual_pred == expected_pred, f"row {i}: prediction drifted ({actual_pred} != {expected_pred})"
