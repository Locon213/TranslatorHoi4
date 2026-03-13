"""Model registry mapping model names to backend factory lambdas."""
from __future__ import annotations

import os
from typing import Dict

from .backends import (
    GoogleFreeBackend,
    G4F_Backend,
    IO_Intelligence_Backend,
    OpenAICompatibleBackend,
    AnthropicBackend,
    GeminiBackend,
    YandexTranslateBackend,
    YandexCloudBackend,
    DeepLBackend,
    FireworksBackend,
    GroqBackend,
    TogetherBackend,
    OllamaBackend,
    MistralBackend,
)


MODEL_REGISTRY: Dict[str, callable] = {
    "Google (free unofficial)": lambda: GoogleFreeBackend(),
    "G4F: API (g4f.dev)": lambda: G4F_Backend(
        api_key=os.environ.get("G4F_API_KEY") or None,
        model=os.environ.get("G4F_MODEL", "gpt-4o"),
        temperature=float(os.environ.get("G4F_TEMP", "0.7")),
        async_mode=(os.environ.get("G4F_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("G4F_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("G4F_RETRIES", "4")),
    ),
    "IO: chat.completions": lambda: IO_Intelligence_Backend(
        api_key=os.environ.get("IO_API_KEY") or None,
        model=os.environ.get("IO_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        base_url=os.environ.get("IO_BASE_URL", "https://api.intelligence.io.solutions/api/v1/"),
        temperature=float(os.environ.get("IO_TEMP", "0.7")),
        async_mode=(os.environ.get("IO_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("IO_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("IO_RETRIES", "4")),
    ),
    "OpenAI Compatible API": lambda: OpenAICompatibleBackend(
        api_key=os.environ.get("OPENAI_API_KEY") or None,
        model=os.environ.get("OPENAI_MODEL", "gpt-4"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
        temperature=float(os.environ.get("OPENAI_TEMP", "0.7")),
        async_mode=(os.environ.get("OPENAI_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("OPENAI_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("OPENAI_RETRIES", "4")),
    ),
    "Anthropic: Claude": lambda: AnthropicBackend(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        temperature=float(os.environ.get("ANTHROPIC_TEMP", "0.7")),
        async_mode=(os.environ.get("ANTHROPIC_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("ANTHROPIC_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("ANTHROPIC_RETRIES", "4")),
    ),
    "Google: Gemini": lambda: GeminiBackend(
        api_key=os.environ.get("GEMINI_API_KEY") or None,
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=float(os.environ.get("GEMINI_TEMP", "0.7")),
        async_mode=(os.environ.get("GEMINI_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("GEMINI_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("GEMINI_RETRIES", "4")),
    ),
    "Yandex Translate": lambda: YandexTranslateBackend(
        api_key=os.environ.get("YANDEX_TRANSLATE_API_KEY") or None,
        iam_token=os.environ.get("YANDEX_IAM_TOKEN") or None,
        folder_id=os.environ.get("YANDEX_FOLDER_ID", ""),
    ),
    "Yandex Cloud": lambda: YandexCloudBackend(
        api_key=os.environ.get("YANDEX_CLOUD_API_KEY") or None,
        model=os.environ.get("YANDEX_CLOUD_MODEL", "aliceai-llm/latest"),
        folder_id=os.environ.get("YANDEX_FOLDER_ID", ""),
        temperature=float(os.environ.get("YANDEX_TEMP", "0.7")),
        async_mode=(os.environ.get("YANDEX_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("YANDEX_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("YANDEX_RETRIES", "4")),
    ),
    "DeepL API": lambda: DeepLBackend(
        api_key=os.environ.get("DEEPL_API_KEY") or None,
    ),
    "Fireworks.ai": lambda: FireworksBackend(
        api_key=os.environ.get("FIREWORKS_API_KEY") or None,
        model=os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
        temperature=float(os.environ.get("FIREWORKS_TEMP", "0.7")),
        async_mode=(os.environ.get("FIREWORKS_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("FIREWORKS_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("FIREWORKS_RETRIES", "4")),
    ),
    "Groq": lambda: GroqBackend(
        api_key=os.environ.get("GROQ_API_KEY") or None,
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=float(os.environ.get("GROQ_TEMP", "0.7")),
        async_mode=(os.environ.get("GROQ_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("GROQ_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("GROQ_RETRIES", "4")),
    ),
    "Together.ai": lambda: TogetherBackend(
        api_key=os.environ.get("TOGETHER_API_KEY") or None,
        model=os.environ.get("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        temperature=float(os.environ.get("TOGETHER_TEMP", "0.7")),
        async_mode=(os.environ.get("TOGETHER_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("TOGETHER_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("TOGETHER_RETRIES", "4")),
    ),
    "Ollama": lambda: OllamaBackend(
        api_key=None,
        model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.environ.get("OLLAMA_TEMP", "0.7")),
        async_mode=(os.environ.get("OLLAMA_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("OLLAMA_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("OLLAMA_RETRIES", "4")),
    ),
    "Mistral AI": lambda: MistralBackend(
        api_key=os.environ.get("MISTRAL_API_KEY") or None,
        model=os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
        temperature=float(os.environ.get("MISTRAL_TEMP", "0.7")),
        async_mode=(os.environ.get("MISTRAL_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("MISTRAL_CONCURRENCY", "6")),
        max_retries=int(os.environ.get("MISTRAL_RETRIES", "4")),
    ),
}
