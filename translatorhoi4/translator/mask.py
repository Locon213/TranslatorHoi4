"""Token masking and unmasking utilities (ported from monolith)."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Regexes
COLOR_CODE_RE = re.compile(r'§!|§.')  # protect §X codes
VAR_RE = re.compile(r'\$[^$\n]+\$')
MACRO_RE = re.compile(r'[^\[\n]+]')  # [Root.GetName]
PERCENT_PLACEHOLDER_RE = re.compile(r'%[-+0-9#.]*[dsf]')
ESC_N_RE = re.compile(r'\\n')
TOKEN_FINDERS = [COLOR_CODE_RE, VAR_RE, MACRO_RE, PERCENT_PLACEHOLDER_RE, ESC_N_RE]
WORD_RE = re.compile(r"[\w\u00C0-\u024F\u0400-\u04FF]+", re.UNICODE)


def mask_tokens(text: str) -> Tuple[str, Dict[str, str], List[str]]:
    mapping: Dict[str, str] = {}
    idx_tokens: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    for rx in TOKEN_FINDERS:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), m.group(0)))
    if not spans:
        return text, {}, []
    spans.sort(key=lambda x: x[0])
    out_parts: List[str] = []
    last = 0
    counter = 0
    for start, end, tok in spans:
        if start < last:
            continue
        out_parts.append(text[last:start])
        key = f"__TKN{counter}__"
        mapping[key] = tok
        idx_tokens.append(tok)
        out_parts.append(key)
        last = end
        counter += 1
    out_parts.append(text[last:])
    masked = "".join(out_parts)
    return masked, mapping, idx_tokens

_TKN_VARIANTS = [
    re.compile(r'__\s*tkn\s*(\d+)\s*__', re.IGNORECASE),
    re.compile(r'_+\s*tkn\s*[_\-\s]*(\d+)\s*_+', re.IGNORECASE),
    re.compile(r'\bT\s*K\s*N\s*[_\- ]\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'\bT\s*K\s*N\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'[`\'"(\[{<]\s*T\s*K\s*N\s*[_\- ]?\s*(\d+)\s*[)`\]}>]', re.IGNORECASE),
]


def _replace_tkn_variants(text: str, idx_tokens: List[str]) -> str:
    def repl(m):
        try:
            i = int(m.group(1))
            return idx_tokens[i] if 0 <= i < len(idx_tokens) else m.group(0)
        except Exception:
            return m.group(0)
    out = text
    for rx in _TKN_VARIANTS:
        out = rx.sub(repl, out)
    return out


def _dunder_aliases_for_tokens(idx_tokens: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for tok in idx_tokens:
        if not tok:
            continue
        if tok.startswith('§') and len(tok) == 2:
            aliases[f"__{tok[1]}__"] = tok
        if tok.startswith('$') and tok.endswith('$') and len(tok) >= 3:
            aliases[f"__{tok[1:-1]}__"] = tok
        if tok.startswith('[') and tok.endswith(']'):
            inner = tok[1:-1]
            if inner and re.fullmatch(r'[A-Za-z0-9_.]+', inner):
                aliases[f"__{inner}__"] = tok
    return aliases


def _replace_dunder_aliases(text: str, idx_tokens: List[str]) -> str:
    aliases = _dunder_aliases_for_tokens(idx_tokens)
    if not aliases:
        return text
    return re.sub(r'__([A-Za-z0-9_.]+)__', lambda m: aliases.get(m.group(0), m.group(0)), text)


def unmask_tokens(text: str, mapping: Dict[str, str], idx_tokens: List[str]) -> str:
    out = text
    for k, v in mapping.items():
        out = out.replace(k, v)
    out = _replace_tkn_variants(out, idx_tokens)
    def restore_leftover(m):
        try:
            i = int(m.group(1))
            return idx_tokens[i] if 0 <= i < len(idx_tokens) else m.group(0)
        except Exception:
            return m.group(0)
    out = re.sub(r'__TKN(\d+)__', restore_leftover, out)
    out = _replace_dunder_aliases(out, idx_tokens)
    out = out.replace("\\ n", "\\n").replace("\\  n", "\\n")
    return out


def count_words_for_stats(text: str) -> int:
    return len(WORD_RE.findall(text))


def _extract_marked(out: str, sid: str) -> str:
    m = re.search(rf"<<SEG {re.escape(sid)}>>(.*?)<<END {re.escape(sid)}>>", out, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"<<SEG [^>]+>>(.*?)<<END [^>]+>>", out, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()
    out2 = re.sub(r"<<\s*(SEG|END)[^>]*>>", "", out)
    return out2.strip()


def _looks_like_http_error(text: str) -> bool:
    t = text.lower()
    return ("error 400" in t) or ("400 bad request" in t) or ("bad request" in t)

