from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence  #, pad_packed_sequence


class SentimentLSTM(nn.Module):
    """Bidirectional LSTM for sentiment classification from text."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int,
        num_classes: int, dropout: float = 0.3, padding_idx: int = 0,
    ) -> None:

        super().__init__()
        self.padding_idx: int = padding_idx
        # 1. Embedding Layer
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # 2. Bidirectional LSTM
        self.lstm: nn.LSTM = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Dropout for regularization
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        # 3. Fully Connected Layer
        self.fc: nn.Linear = nn.Linear(hidden_dim * 2, num_classes)
 
    def forward(self, x: Tensor) -> Tensor:
        """Runs the model's forward pass."""

        # compute the real length of each sequence in the batch
        lengths: Tensor = (x != self.padding_idx).sum(dim=1)
        lengths = lengths.clamp(min=1)
        embedded: Tensor = self.embedding(x)

        # Pack the sequence before feeding it into the LSTM while skipping padding internally
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _packed_output, (hidden, _cell) = self.lstm(packed_embedded)
        # Unpacking for attention:
        # lstm_out, out_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # Concatenate the last forward and backward hidden states
        # hidden[-2] -> last layer, forward direction; hidden[-1] -> last layer, backward direction
        hidden_cat: Tensor = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        dropped: Tensor = self.dropout(hidden_cat)
        out: Tensor = self.fc(dropped)
        return out
    
    def export_config(self) -> dict:
        """Returns the model's configuration as a dictionary."""
        return {
            "vocab_size": self.embedding.num_embeddings,
            "embedding_dim": self.embedding.embedding_dim,
            "hidden_dim": self.lstm.hidden_size,
            "num_classes": self.fc.out_features,
            "padding_idx": self.padding_idx
        }