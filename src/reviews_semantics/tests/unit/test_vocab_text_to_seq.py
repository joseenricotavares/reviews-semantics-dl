from reviews_semantics.data.vocab import PAD_TOKEN, UNK_TOKEN, TextPreprocessor, build_vocab, text_to_seq

CORPUS = ["produto otimo chegou rapido", "produto ruim chegou quebrado", "otimo atendimento"]


def test_build_vocab_reserves_pad_and_unk_at_index_0_and_1():
    vocab = build_vocab(CORPUS)
    assert vocab[PAD_TOKEN] == 0
    assert vocab[UNK_TOKEN] == 1


def test_build_vocab_orders_by_frequency():
    vocab = build_vocab(CORPUS)
    # "produto" and "chegou" each appear twice, everything else once.
    assert vocab["produto"] < vocab["ruim"]
    assert vocab["chegou"] < vocab["quebrado"]


def test_build_vocab_respects_max_words_budget():
    vocab = build_vocab(CORPUS, max_words=4)
    assert len(vocab) == 4  # <PAD>, <UNK>, + 2 most frequent words


def test_text_to_seq_maps_unknown_words_to_unk():
    vocab = build_vocab(CORPUS)
    seq = text_to_seq("produto nunca_visto", vocab, max_len=5)
    assert seq[0] == vocab["produto"]
    assert seq[1] == vocab[UNK_TOKEN]


def test_text_to_seq_pads_short_sequences():
    vocab = build_vocab(CORPUS)
    seq = text_to_seq("otimo", vocab, max_len=5)
    assert len(seq) == 5
    assert seq[1:] == [vocab[PAD_TOKEN]] * 4


def test_text_to_seq_truncates_long_sequences():
    vocab = build_vocab(CORPUS)
    seq = text_to_seq("produto otimo chegou rapido produto ruim", vocab, max_len=3)
    assert len(seq) == 3


def test_text_to_seq_handles_empty_text():
    vocab = build_vocab(CORPUS)
    seq = text_to_seq("", vocab, max_len=4)
    assert seq == [vocab[PAD_TOKEN]] * 4


def test_preprocessor_round_trips_through_dict():
    preprocessor = TextPreprocessor.build(CORPUS, max_len=10, max_vocab_size=50)
    restored = TextPreprocessor.from_dict(preprocessor.to_dict())

    assert restored.vocab == preprocessor.vocab
    assert restored.max_len == preprocessor.max_len
    assert restored.encode("produto otimo") == preprocessor.encode("produto otimo")
