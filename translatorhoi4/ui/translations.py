"""Runtime loader for TranslatorHoi4 UI translations."""
from __future__ import annotations

from importlib import import_module
from types import MappingProxyType


_TRANSLATION_CACHE: dict[str, dict[str, str]] = {}


def _load_language(lang_code: str) -> dict[str, str]:
    normalized = (lang_code or "english").lower()
    if normalized == "english":
        return {}
    if normalized not in _TRANSLATION_CACHE:
        try:
            module = import_module(f"{__package__}.locales.{normalized}")
            translations = getattr(module, "TRANSLATIONS", {})
            _TRANSLATION_CACHE[normalized] = dict(translations)
        except Exception:
            _TRANSLATION_CACHE[normalized] = {}
    return _TRANSLATION_CACHE[normalized]


def available_translations() -> MappingProxyType[str, dict[str, str]]:
    """Return loaded translation dictionaries for diagnostics/tests."""
    return MappingProxyType(_TRANSLATION_CACHE)


def translate_text(text: str, lang_code: str) -> str:
    """Translate UI text, falling back to the original English text."""
    if not isinstance(text, str) or not text:
        return text
    return _load_language(lang_code).get(text, text)


def set_translations(new_translations: dict[str, dict[str, str]]) -> None:
    """Add or update translations dynamically."""
    for lang_code, translations in new_translations.items():
        normalized = (lang_code or "english").lower()
        _TRANSLATION_CACHE.setdefault(normalized, {}).update(translations)
