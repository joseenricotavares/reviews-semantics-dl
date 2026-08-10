from dlkit.artifacts import ArtifactBundle
from reviews_semantics.labels import STAR_SCHEMA
from reviews_semantics.paradigms.bilstm.predictor import BiLSTMPredictor


def test_load_reconstructs_a_working_predictor(bilstm_bundle_dir):
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(bilstm_bundle_dir))
    scores = predictor.predict(["Produto ótimo, chegou rápido!"])
    assert len(scores) == 1
    assert scores[0] in STAR_SCHEMA.classes


def test_predict_applies_clean_text_before_encoding(bilstm_bundle_dir):
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(bilstm_bundle_dir))
    raw = predictor.predict(["PRODUTO ÓTIMO!!! Chegou rápido :)"])
    pre_cleaned = predictor.predict(["produto ótimo chegou rápido"])
    assert raw == pre_cleaned


def test_predict_is_deterministic(bilstm_bundle_dir):
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(bilstm_bundle_dir))
    texts = ["produto excelente", "produto pessimo", "entrega no prazo"]
    assert predictor.predict(texts) == predictor.predict(texts)


def test_label_schema_matches_star_schema(bilstm_bundle_dir):
    predictor = BiLSTMPredictor.load(ArtifactBundle.read(bilstm_bundle_dir))
    assert predictor.label_schema == STAR_SCHEMA
