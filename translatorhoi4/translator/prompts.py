"""Prompt helpers."""
from __future__ import annotations

import hashlib


def system_prompt(src_lang: str, dst_lang: str) -> str:
    return (
        "You are a specialised game localisation engine. "
        f"Translate from {src_lang} to {dst_lang}. "
        "Keep tokens intact: $VARS$, [Scripted.Macros], and \\n. "
        "Return ONLY the translated text between the SAME markers."
    )


def seg_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def wrap_with_markers(text: str, sid: str) -> str:
    return f"<<SEG {sid}>>\n{text}\n<<END {sid}>>"
