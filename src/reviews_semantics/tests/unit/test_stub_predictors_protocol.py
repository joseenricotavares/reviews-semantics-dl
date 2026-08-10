from dlkit.inference import Predictor
from reviews_semantics.paradigms.feature_extraction.predictor import FeatureExtractionPredictor
from reviews_semantics.paradigms.feature_extraction.trainer import TransformerFeatureExtractor
from reviews_semantics.paradigms.transformer_finetune.predictor import TransformerPredictor


def test_transformer_predictor_satisfies_the_predictor_protocol():
    predictor = TransformerPredictor(model=object(), tokenizer=object(), device="cpu")
    assert isinstance(predictor, Predictor)


def test_feature_extraction_predictor_satisfies_the_predictor_protocol():
    extractor = TransformerFeatureExtractor.__new__(TransformerFeatureExtractor)
    predictor = FeatureExtractionPredictor(extractor)
    assert isinstance(predictor, Predictor)
