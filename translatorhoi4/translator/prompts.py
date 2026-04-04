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


def parse_batch_response(response: str, expected_keys: list = None) -> dict:
    """Parse batch translation response into key-value pairs.
    
    Args:
        response: Raw response from the model
        expected_keys: List of keys we expect to find (for validation)
    
    Returns:
        Dictionary mapping keys to their translations
    """
    translations = {}
    lines = response.strip().split('\n')
    
    # Remove common markdown artifacts
    cleaned_lines = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        # Skip code block markers
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            cleaned_lines.append(line)
    
    for line in cleaned_lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that look like markdown or explanatory text
        if line.startswith('#') or line.startswith('*') or line.startswith('- '):
            continue
        if line.lower().startswith(('here', 'sure', 'okay', 'ok ', 'note', 'important')):
            continue
            
        # Handle different formats: "KEY: TRANSLATION" or "KEY TRANSLATION"
        if ':' in line:
            key, translation = line.split(':', 1)
            key = key.strip()
            translation = translation.strip()
            
            # Validate that key looks like a valid game key (uppercase, underscores, etc.)
            # Game keys typically match patterns like: STATE_NAME, PARTY_DESC, etc.
            if key and translation and (len(key) > 1 and not key[0].islower()):
                translations[key] = translation
        else:
            # Fallback for cases where colon might be missing - skip these
            # as they're likely part of the translation text
            continue

    # If we have expected keys and got significantly fewer, log a warning
    if expected_keys and len(translations) < len(expected_keys):
        missing = set(expected_keys) - set(translations.keys())
        if missing:
            # Try to find missing keys with more lenient parsing
            for missing_key in missing:
                # Search for the key in the original response
                for line in lines:
                    if line.strip().startswith(f"{missing_key}:"):
                        _, trans = line.strip().split(':', 1)
                        translations[missing_key] = trans.strip()
                        break

    return translations


def seg_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def wrap_with_markers(text: str, sid: str) -> str:
    return f"<<SEG {sid}>>\n{text}\n<<END {sid}>>"
