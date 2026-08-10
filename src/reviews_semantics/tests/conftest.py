from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from dlkit.artifacts import ArtifactBundle, BundleFile
from reviews_semantics.data.vocab import TextPreprocessor
from reviews_semantics.labels import STAR_SCHEMA
from reviews_semantics.paradigms.bilstm.model import SentimentLSTM


@pytest.fixture
def bilstm_bundle_dir(tmp_path: Path) -> Path:
    """Writes a tiny, fast-to-build BiLSTM `ArtifactBundle` for tests that
    need a real, loadable model without depending on the full committed
    `models/bilstm-baseline/` checkpoint."""
    preprocessor = TextPreprocessor.build(
        ["produto otimo chegou rapido", "produto ruim chegou quebrado"], max_len=10, max_vocab_size=50
    )
    model = SentimentLSTM(
        vocab_size=len(preprocessor.vocab),
        embedding_dim=8,
        hidden_dim=16,
        num_classes=5,
        padding_idx=preprocessor.vocab["<PAD>"],
    )

    root = tmp_path / "bundle"
    ArtifactBundle.write(
        root,
        flavor="bilstm",
        label_schema=STAR_SCHEMA,
        files={
            "weights": BundleFile("state_dict.pt", lambda p: torch.save(model.state_dict(), p)),
            "model_config": BundleFile(
                "model.json", lambda p: p.write_text(json.dumps(model.export_config()))
            ),
            "preprocessor": BundleFile(
                "preprocessor.json",
                lambda p: p.write_text(
                    json.dumps(preprocessor.to_dict(), ensure_ascii=False), encoding="utf-8"
                ),
            ),
        },
    )
    return root
