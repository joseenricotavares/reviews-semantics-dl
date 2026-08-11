from importlib.metadata import version as _pkg_version

from reviews_semantics.data import TextPreprocessor, clean_text
from reviews_semantics.evaluation import StarEvaluation, evaluate_stars
from reviews_semantics.labels import STAR_SCHEMA
from reviews_semantics.paradigms.bilstm import BiLSTMPredictor, ClassifierTrainer, SentimentLSTM
from reviews_semantics.registry import load_default_model, load_default_predictor
from reviews_semantics.serving_config import Settings
from reviews_semantics.training_config import TrainingConfig

__version__ = _pkg_version("reviews-semantics")

__all__ = [
    "__version__",
    "Settings",
    "clean_text",
    "TextPreprocessor",
    "STAR_SCHEMA",
    "TrainingConfig",
    "SentimentLSTM",
    "ClassifierTrainer",
    "BiLSTMPredictor",
    "StarEvaluation",
    "evaluate_stars",
    "load_default_predictor",
    "load_default_model",
]
