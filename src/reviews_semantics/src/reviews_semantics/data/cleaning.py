from __future__ import annotations

import re

_NON_LETTER_PATTERN = re.compile(r"[^a-záéíóúâêôãõç\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _NON_LETTER_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
