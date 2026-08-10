import torch
from reviews_semantics.paradigms.bilstm.model import SentimentLSTM


def _make_model() -> SentimentLSTM:
    return SentimentLSTM(vocab_size=50, embedding_dim=8, hidden_dim=16, num_classes=5, padding_idx=0)


def test_forward_output_shape_matches_batch_and_num_classes():
    model = _make_model()
    batch = torch.randint(1, 50, (4, 10))
    batch[:, 6:] = 0  # simulate padding

    logits = model(batch)

    assert logits.shape == (4, 5)


def test_forward_handles_all_padding_row_without_crashing():
    # `lengths.clamp(min=1)` exists precisely to keep pack_padded_sequence
    # from choking on an all-<PAD> sequence.
    model = _make_model()
    batch = torch.zeros((2, 10), dtype=torch.long)

    logits = model(batch)

    assert logits.shape == (2, 5)


def test_export_config_round_trips_into_an_equivalent_model():
    model = _make_model()
    config = model.export_config()

    rebuilt = SentimentLSTM(**config)

    assert rebuilt.embedding.num_embeddings == model.embedding.num_embeddings
    assert rebuilt.embedding.embedding_dim == model.embedding.embedding_dim
    assert rebuilt.lstm.hidden_size == model.lstm.hidden_size
    assert rebuilt.fc.out_features == model.fc.out_features
    assert rebuilt.padding_idx == model.padding_idx
