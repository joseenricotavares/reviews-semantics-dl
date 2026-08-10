from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DEFAULT_MAX_LEN = 50
DEFAULT_MAX_VOCAB_SIZE = 10000


def build_vocab(texts: Iterable[str], max_words: int = DEFAULT_MAX_VOCAB_SIZE) -> dict[str, int]:
    """Builds a word -> index vocabulary from whitespace-tokenized `texts`."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_words - 2):
        vocab[word] = len(vocab)
    return vocab


def text_to_seq(text: str, vocab: dict[str, int], max_len: int = DEFAULT_MAX_LEN) -> list[int]:
    """Tokenizes `text` by whitespace, maps each token through `vocab`
    (falling back to `<UNK>`), then pads/truncates to `max_len`."""
    tokens = text.split()
    seq = [vocab.get(word, vocab[UNK_TOKEN]) for word in tokens]
    if len(seq) < max_len:
        seq = seq + [vocab[PAD_TOKEN]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


@dataclass(slots=True)
class TextPreprocessor:
    """Word-level vocabulary + fixed-length encoding, serializable to/from a
    plain dict so it can round-trip through an `ArtifactBundle`."""

    vocab: dict[str, int]
    max_len: int

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_len: int = DEFAULT_MAX_LEN,
        max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    ) -> TextPreprocessor:
        return cls(vocab=build_vocab(texts, max_vocab_size), max_len=max_len)

    def encode(self, text: str) -> list[int]:
        return text_to_seq(text, self.vocab, self.max_len)

    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, data: dict) -> TextPreprocessor:
        return cls(vocab=dict(data["vocab"]), max_len=int(data["max_len"]))
