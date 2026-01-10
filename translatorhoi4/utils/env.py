"""Environment variable and .env file support for sensitive data."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Try to import python-dotenv, fall back to basic parsing if not available
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False


class EnvLoader:
    """Load environment variables from .env file with priority support."""
    
    _instance: Optional["EnvLoader"] = None
    _loaded: bool = False
    
    def __new__(cls) -> "EnvLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._loaded:
            return
        self._loaded = True
        self._env_vars: Dict[str, str] = {}
        self._load_env_file()
    
    def _load_env_file(self):
        """Load .env file from multiple possible locations."""
        # Try multiple locations for .env file
        possible_paths = [
            Path(".env"),
            Path(".env.local"),
            Path(__file__).parent.parent.parent / ".env",
            Path.home() / ".translatorhoi4" / ".env",
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                if _HAS_DOTENV:
                    load_dotenv(env_path, override=True)
                else:
                    self._parse_env_file(env_path)
                # Store the last loaded file path
                self._env_file_path = env_path
    
    def _parse_env_file(self, path: Path):
        """Basic .env file parser for when python-dotenv is not available."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    self._env_vars[key] = value
                    os.environ[key] = value
        except Exception:
            pass
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable, checking .env first, then os.environ."""
        # Check our cached .env vars first
        if key in self._env_vars:
            return self._env_vars[key]
        # Fall back to os.environ
        return os.environ.get(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean."""
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get environment variable as integer."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(float(value))
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get environment variable as float."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    def set(self, key: str, value: str, persist: bool = False):
        """Set environment variable."""
        os.environ[key] = value
        self._env_vars[key] = value
        # Note: persist=True would require writing back to .env file
    
    @property
    def env_file_path(self) -> Optional[Path]:
        """Get the path to the loaded .env file."""
        return getattr(self, "_env_file_path", None)


# Global instance
env = EnvLoader()


# --- API Key Environment Variable Names ---
API_KEY_ENV_VARS = {
    "g4f": ["G4F_API_KEY", "G4F_KEY"],
    "openai": ["OPENAI_API_KEY", "OPENAI_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY"],
    "gemini": ["GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"],
    "io": ["IO_API_KEY", "IO_KEY"],
}


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider from environment variables."""
    env_vars = API_KEY_ENV_VARS.get(provider.lower(), [])
    for env_var in env_vars:
        key = env.get(env_var)
        if key:
            return key
    return None


def set_api_key(provider: str, key: str) -> None:
    """Set API key for a specific provider."""
    env_vars = API_KEY_ENV_VARS.get(provider.lower(), [])
    if env_vars:
        env.set(env_vars[0], key)


# --- Cost Tracking Environment Variables ---
COST_CONFIG_ENV_VARS = {
    "g4f_input_cost": "G4F_COST_INPUT_PER_MILLION",
    "g4f_output_cost": "G4F_COST_OUTPUT_PER_MILLION",
    "openai_input_cost": "OPENAI_COST_INPUT_PER_MILLION",
    "openai_output_cost": "OPENAI_COST_OUTPUT_PER_MILLION",
    "anthropic_input_cost": "ANTHROPIC_COST_INPUT_PER_MILLION",
    "anthropic_output_cost": "ANTHROPIC_COST_OUTPUT_PER_MILLION",
    "gemini_input_cost": "GEMINI_COST_INPUT_PER_MILLION",
    "gemini_output_cost": "GEMINI_COST_OUTPUT_PER_MILLION",
    "io_input_cost": "IO_COST_INPUT_PER_MILLION",
    "io_output_cost": "IO_COST_OUTPUT_PER_MILLION",
    "currency": "COST_CURRENCY",
}


def get_cost_config(provider: str, is_input: bool = True) -> float:
    """Get cost per million tokens for a provider."""
    env_key = f"{provider}_input_cost" if is_input else f"{provider}_output_cost"
    env_var = COST_CONFIG_ENV_VARS.get(env_key)
    if env_var:
        return env.get_float(env_var, 0.0)
    return 0.0


def get_cost_currency() -> str:
    """Get the currency for cost display."""
    return env.get(COST_CONFIG_ENV_VARS.get("currency", "currency"), "USD")
