"""Translation backends package — re-exports all backends for backward compatibility."""
from __future__ import annotations

# Base class and helpers
from .base import (
    TranslationBackend,
    _extract_text,
    _cleanup_llm_output,
    _strip_model_noise,
)

# Individual backends
from .google import GoogleFreeBackend
from .g4f import G4F_Backend
from .io_intelligence import IO_Intelligence_Backend
from .anthropic_backend import AnthropicBackend
from .gemini import GeminiBackend
from .yandex import YandexTranslateBackend, YandexCloudBackend
from .deepl import DeepLBackend
from .fireworks import FireworksBackend
from .groq import GroqBackend
from .together import TogetherBackend
from .ollama import OllamaBackend
from .openai_compat import OpenAICompatibleBackend
from .mistral import MistralBackend
from .nvidia_nim import NvidiaNIMBackend

__all__ = [
    # Base
    "TranslationBackend",
    "_extract_text",
    "_cleanup_llm_output",
    "_strip_model_noise",
    # Backends
    "GoogleFreeBackend",
    "G4F_Backend",
    "IO_Intelligence_Backend",
    "AnthropicBackend",
    "GeminiBackend",
    "YandexTranslateBackend",
    "YandexCloudBackend",
    "DeepLBackend",
    "FireworksBackend",
    "GroqBackend",
    "TogetherBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "MistralBackend",
    "NvidiaNIMBackend",
]
