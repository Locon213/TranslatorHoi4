"""Preset and settings load/save utilities."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


# Default settings file path
SETTINGS_FILE = "translatorhoi4_settings.json"


def get_default_settings_path() -> str:
    """Get the default path for the settings file."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), SETTINGS_FILE)


def save_settings(data: Dict[str, Any], path: Optional[str] = None) -> bool:
    """Save settings to a JSON file.
    
    Args:
        data: Settings dictionary to save
        path: Optional custom path, otherwise uses default
    
    Returns:
        True if successful, False otherwise
    """
    if path is None:
        path = get_default_settings_path()
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save settings: {e}")
        return False


def load_settings(path: Optional[str] = None) -> Dict[str, Any]:
    """Load settings from a JSON file.
    
    Args:
        path: Optional custom path, otherwise uses default
    
    Returns:
        Settings dictionary, empty dict if file doesn't exist or error
    """
    if path is None:
        path = get_default_settings_path()
    
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load settings: {e}")
    return {}


def save_preset(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_preset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
