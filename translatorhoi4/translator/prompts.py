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

def batch_system_prompt(src_lang: str, dst_lang: str) -> str:
    return (
        "You are a specialised game localisation engine. "
        f"Translate from {src_lang} to {dst_lang}. "
        "Keep tokens intact: $VARS$, [Scripted.Macros], and \\n. "
        "Return ONLY the translated text in the format 'KEY: TRANSLATION' without any markdown, code blocks, or additional formatting. "
        "Each translation should be on a separate line. "
        "Do not use any backticks, triple quotes, or markdown syntax. "
        "Output plain text only."
    )

def batch_wrap_with_markers(batch_data: dict) -> str:
    """Wrap batch data with markers for batch translation."""
    lines = []
    for key, text in batch_data.items():
        sid = seg_id(text)
        lines.append(f"{key}: <<SEG {sid}>>")
        lines.append(text)
        lines.append(f"<<END {sid}>>")
    return "\n".join(lines)


def parse_batch_response(response: str) -> dict:
    """Parse batch translation response into key-value pairs."""
    translations = {}
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Handle different formats: "KEY: TRANSLATION" or "KEY TRANSLATION"
        if ':' in line:
            key, translation = line.split(':', 1)
            key = key.strip()
            translation = translation.strip()
        else:
            # Fallback for cases where colon might be missing
            parts = line.split(' ', 1)
            if len(parts) >= 2:
                key = parts[0].strip()
                translation = ' '.join(parts[1:]).strip()
            else:
                continue
        
        if key and translation:
            translations[key] = translation
    
    return translations


def seg_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def wrap_with_markers(text: str, sid: str) -> str:
    return f"<<SEG {sid}>>\n{text}\n<<END {sid}>>"
