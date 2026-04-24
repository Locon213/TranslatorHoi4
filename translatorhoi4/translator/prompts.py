"""Prompt helpers for AI translation."""
from __future__ import annotations

import json
import hashlib
import re
from typing import Optional

from .game_profiles import get_game_profile_instruction


def system_prompt(src_lang: str, dst_lang: str, game_id: Optional[str] = None, mod_theme: Optional[str] = None) -> str:
    """Generate system prompt for single-segment translation.
    
    Args:
        src_lang: Source language name
        dst_lang: Destination language name
        game_id: Game identifier for context-specific translation
        mod_theme: Optional custom mod theme
    
    Returns:
        System prompt string optimized for game localization
    """
    # Game-specific instruction with optional mod theme
    game_instruction = get_game_profile_instruction(game_id, mod_theme) if game_id else ""
    
    # Base prompt with improved structure
    base = (
        "You are a professional video game localization translator.\n"
        f"Translate the following text from {src_lang} to {dst_lang}.\n"
    )
    
    # Add game context if available
    if game_instruction:
        base += f"\n{game_instruction}\n"
    
    # Critical rules
    rules = (
        "\nCRITICAL RULES:\n"
        "1. Preserve ALL tokens EXACTLY: $VARS$, [Scripted.Macros], \\n, and similar placeholders\n"
        "2. Return ONLY the translated text between the SAME <<SEG>> markers - nothing else\n"
        "3. Do NOT add explanations, notes, comments, or extra text\n"
        "4. Preserve original formatting, capitalization style, and punctuation patterns\n"
        "5. Translate accurately while maintaining the original meaning, tone, and context\n"
        "6. For proper nouns (names, places, titles), use established game translations if they exist\n"
        "7. If text is empty, unclear, or already translated, return it unchanged\n"
        "8. Never translate content inside $ $ or [ ] brackets\n\n"
        "OUTPUT: Only the translated text with markers, nothing else."
    )
    
    return base + rules


def batch_system_prompt(src_lang: str, dst_lang: str, game_id: Optional[str] = None, mod_theme: Optional[str] = None) -> str:
    """Generate system prompt for batch translation mode.
    
    Args:
        src_lang: Source language name
        dst_lang: Destination language name
        game_id: Game identifier for context-specific translation
        mod_theme: Optional custom mod theme
    
    Returns:
        System prompt string optimized for batch translation
    """
    # Game-specific instruction with optional mod theme
    game_instruction = get_game_profile_instruction(game_id, mod_theme) if game_id else ""
    
    # Base prompt
    base = (
        "You are a professional video game localization translator working in batch mode.\n"
        f"Translate multiple text segments from {src_lang} to {dst_lang}.\n"
    )
    
    # Add game context
    if game_instruction:
        base += f"\n{game_instruction}\n"
    
    # Batch-specific rules
    rules = (
        "\nCRITICAL RULES:\n"
        "1. Preserve ALL tokens EXACTLY: $VARS$, [Scripted.Macros], \\n, and similar placeholders\n"
        "2. Translate each segment between its <<SEG>> and <<END>> markers\n"
        "3. Return ONLY translations in the format 'B001: TRANSLATION' - one per line\n"
        "4. Do NOT use markdown, code blocks, backticks, or triple quotes\n"
        "5. Do NOT add explanations, notes, headers, or extra text\n"
        "6. Preserve original formatting, capitalization, and punctuation\n"
        "7. Translate accurately while maintaining meaning, tone, and context\n"
        "8. For proper nouns, use established game translations\n"
        "9. If text is empty or unclear, return it unchanged\n"
        "10. Never translate content inside $ $ or [ ] brackets\n\n"
        "OUTPUT FORMAT:\n"
        "B001: translation1\n"
        "B002: translation2\n"
        "B003: translation3\n\n"
        "Use the exact B-number ids from the input. Plain text only - no markdown, no YAML, no code blocks."
    )
    
    return base + rules

def batch_wrap_with_markers(batch_data: dict) -> str:
    """Wrap batch data with markers for batch translation."""
    lines = []
    for key, text in batch_data.items():
        sid = seg_id(text)
        lines.append(f"{key}: <<SEG {sid}>>")
        lines.append(text)
        lines.append(f"<<END {sid}>>")
    return "\n".join(lines)


def parse_batch_response(response: str, expected_keys: list = None, aliases: dict = None) -> dict:
    """Parse batch translation response into key-value pairs.
    
    Args:
        response: Raw response from the model
        expected_keys: List of keys we expect to find (for validation)
        aliases: Optional mapping of alternate/original keys to expected keys
    
    Returns:
        Dictionary mapping keys to their translations
    """
    translations = {}
    expected_keys = [str(key) for key in (expected_keys or [])]
    expected_set = set(expected_keys)
    alias_map = {
        str(alias): str(target)
        for alias, target in (aliases or {}).items()
        if alias is not None and target is not None
    }
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
    
    cleaned_response = "\n".join(cleaned_lines).strip()

    def normalize_key(raw_key: str) -> str:
        key = raw_key.strip().strip("`'\"")
        key = re.sub(r"^\s*(?:[-*]\s+|\d+[\.)]\s*)", "", key).strip()
        return key

    def resolve_key(raw_key: str) -> str:
        key = normalize_key(raw_key)
        return alias_map.get(key, key)

    def normalize_translation(raw_translation: str) -> str:
        translation = raw_translation.strip()
        translation = re.sub(r"<<\s*SEG\b[^>]*>>", "", translation, flags=re.IGNORECASE).strip()
        translation = re.sub(r"<<\s*END\b[^>]*>>", "", translation, flags=re.IGNORECASE).strip()
        version_match = re.fullmatch(r'\s*\d+\s+"(.*)"\s*(?:#.*)?', translation)
        if version_match:
            translation = version_match.group(1).strip()
        if len(translation) >= 2 and translation[0] == translation[-1] and translation[0] in {'"', "'"}:
            translation = translation[1:-1].strip()
        return translation

    def accept_translation(key: str, translation: str) -> None:
        key = resolve_key(key)
        translation = normalize_translation(translation)
        if not key or not translation:
            return
        if expected_set and key not in expected_set:
            return
        translations[key] = translation

    if cleaned_response:
        try:
            parsed_json = json.loads(cleaned_response)
            if isinstance(parsed_json, dict):
                for key, value in parsed_json.items():
                    accept_translation(str(key), str(value))
        except Exception:
            pass

    # Some models mirror the input marker block instead of returning KEY: translation.
    # Parse those blocks before line-based parsing so the real translated body wins.
    keys_for_blocks = expected_keys + list(alias_map.keys()) if expected_keys else [
        normalize_key(line.split(":", 1)[0])
        for line in cleaned_lines
        if ":" in line
    ]
    for key in keys_for_blocks:
        if not key:
            continue
        block_re = re.compile(
            rf"(?ms)^\s*(?:[-*]\s+|\d+[\.)]\s*)?{re.escape(key)}\s*:\s*"
            r"<<\s*SEG\b[^>]*>>\s*(.*?)\s*<<\s*END\b[^>]*>>"
        )
        match = block_re.search(cleaned_response)
        if match:
            accept_translation(key, match.group(1))

    for line in cleaned_lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that look like markdown or explanatory text
        if line.startswith('#') or line.startswith('*') or (line.startswith('- ') and ':' not in line):
            continue
        if re.fullmatch(r"l_[A-Za-z_]+:", line):
            continue
        if re.fullmatch(r"<<\s*(SEG|END)\b[^>]*>>", line, flags=re.IGNORECASE):
            continue
        if line.lower().startswith(('here', 'sure', 'okay', 'ok ', 'note', 'important')):
            continue
            
        # Handle different formats: "KEY: TRANSLATION" or "KEY TRANSLATION"
        if ':' in line:
            key, translation = line.split(':', 1)
            accept_translation(key, translation)
        else:
            sep_match = re.match(r"^([A-Za-z0-9_.-]+)\s+(?:=>|->|-)\s+(.+)$", line)
            if sep_match:
                accept_translation(sep_match.group(1), sep_match.group(2))

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
                        accept_translation(missing_key, trans)
                        break

    return translations


def seg_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def wrap_with_markers(text: str, sid: str) -> str:
    return f"<<SEG {sid}>>\n{text}\n<<END {sid}>>"
