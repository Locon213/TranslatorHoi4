"""Helpers for UI language selectors."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

from ..parsers.paradox_yaml import LANG_NAME_LIST, get_native_language_name


FLAG_ICON_FILES = {
    "english": "us.svg",
    "russian": "ru.svg",
    "german": "de.svg",
    "french": "fr.svg",
    "spanish": "es.svg",
    "braz_por": "br.svg",
    "polish": "pl.svg",
    "japanese": "jp.svg",
    "korean": "kr.svg",
    "simp_chinese": "cn.svg",
}


def _resource_path(rel_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel_path
    return Path(__file__).resolve().parent.parent.parent / rel_path


def language_icon(lang_code: str) -> QIcon:
    file_name = FLAG_ICON_FILES.get(lang_code)
    if not file_name:
        return QIcon()
    return QIcon(str(_resource_path(f"assets/flags/{file_name}")))


def populate_language_combo(combo) -> None:
    combo.clear()
    for code in LANG_NAME_LIST:
        combo.addItem(get_native_language_name(code), icon=language_icon(code), userData=code)
