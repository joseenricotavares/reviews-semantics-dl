import csv

import pytest
from dlkit.artifacts import ArtifactBundle
from reviews_semantics.cli.batch import predict_csv
from reviews_semantics.paradigms.bilstm.predictor import BiLSTMPredictor


def _write_csv(path, texts):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review_comment_message"])
        writer.writeheader()
        for text in texts:
            writer.writerow({"review_comment_message": text})


def test_predict_csv_adds_a_score_column(tmp_path, bilstm_bundle_dir):
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(bilstm_bundle_dir))
    input_path = tmp_path / "input.csv"
    _write_csv(input_path, ["produto otimo", "produto ruim"])

    output_path = tmp_path / "output.csv"
    count = predict_csv(input_path, output_path, predictor=predictor, chunk_size=1)

    assert count == 2
    with output_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(1 <= int(row["predicted_score"]) <= 5 for row in rows)


def test_predict_csv_matches_direct_predictor_calls(tmp_path, bilstm_bundle_dir):
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(bilstm_bundle_dir))
    texts = ["entrega rapida", "produto quebrado", "otimo atendimento"]
    input_path = tmp_path / "input.csv"
    _write_csv(input_path, texts)

    output_path = tmp_path / "output.csv"
    predict_csv(input_path, output_path, predictor=predictor)

    with output_path.open(newline="", encoding="utf-8") as f:
        csv_scores = [int(row["predicted_score"]) for row in csv.DictReader(f)]

    assert csv_scores == predictor.predict(texts)


def test_predict_csv_raises_on_missing_text_column(tmp_path, bilstm_bundle_dir):
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(bilstm_bundle_dir))
    input_path = tmp_path / "input.csv"
    with input_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["some_other_column"])
        writer.writeheader()
        writer.writerow({"some_other_column": "x"})

    with pytest.raises(ValueError):
        predict_csv(input_path, tmp_path / "out.csv", predictor=predictor)
