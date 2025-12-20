"""Glossary helpers (ported from monolith)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Glossary:
    protect: List[str]
    replacements: Dict[str, str]

    @staticmethod
    def load_csv(path: str) -> "Glossary":
        protect: List[str] = []
        repl: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            sniff = f.read(4096); f.seek(0)
            dialect = csv.Sniffer().sniff(sniff, delimiters=",;")
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                mode = (row.get("mode") or "").strip().lower()
                src = (row.get("src") or "").strip()
                dst = (row.get("dst") or "").strip()
                if not mode or not src:
                    continue
                if mode == "protect":
                    protect.append(src)
                elif mode == "replace" and dst:
                    repl[src] = dst
        return Glossary(protect, repl)


def _mask_glossary(text: str, gl: Glossary) -> Tuple[str, Dict[str, str]]:
    if not gl.protect:
        return text, {}
    mapping: Dict[str, str] = {}
    masked = text
    for i, phrase in enumerate(sorted(gl.protect, key=len, reverse=True)):
        if not phrase:
            continue
        key = f"__GLS{i}__"
        if phrase in masked:
            mapping[key] = phrase
            masked = masked.replace(phrase, key)
    return masked, mapping


def _unmask_glossary(text: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def _apply_replacements(text: str, gl: Glossary) -> str:
    if not gl.replacements:
        return text
    out = text
    for src, dst in gl.replacements.items():
        if not src:
            continue
        if __import__('re').match(r"^[\w\-]+$", src, flags=__import__('re').UNICODE):
            out = __import__('re').sub(rf"\b{__import__('re').escape(src)}\b", dst, out)
        else:
            out = out.replace(src, dst)
    return out
