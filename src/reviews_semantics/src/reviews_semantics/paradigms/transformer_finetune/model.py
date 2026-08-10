from __future__ import annotations


def load_model_and_tokenizer(model_id: str, num_classes: int):
    """Loads a HuggingFace sequence-classification model + its tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)
    return tokenizer, model
