"""Job configuration dataclass for translation engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class JobConfig:
    src_dir: str
    out_dir: str
    src_lang: str
    dst_lang: str
    model_key: str
    temperature: float
    in_place: bool
    skip_existing: bool
    strip_md: bool
    batch_size: int
    rename_files: bool
    files_concurrency: int
    key_skip_regex: Optional[str]
    cache_path: Optional[str]
    cache_type: str = "sqlite"
    glossary_path: Optional[str] = None
    prev_loc_dir: Optional[str] = None
    reuse_prev_loc: bool = False
    mark_loc_flag: bool = False
    g4f_model: Optional[str] = None
    g4f_api_key: Optional[str] = None
    g4f_async: bool = True
    g4f_concurrency: int = 6
    io_model: Optional[str] = None
    io_api_key: Optional[str] = None
    io_base_url: Optional[str] = None
    io_async: bool = True
    io_concurrency: int = 6
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_async: bool = True
    openai_concurrency: int = 6
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None
    anthropic_async: bool = True
    anthropic_concurrency: int = 6
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_async: bool = True
    gemini_concurrency: int = 6
    yandex_translate_api_key: Optional[str] = None
    yandex_iam_token: Optional[str] = None
    yandex_folder_id: str = ""
    yandex_cloud_api_key: Optional[str] = None
    yandex_cloud_model: Optional[str] = None
    yandex_async: bool = True
    yandex_concurrency: int = 6
    deepl_api_key: Optional[str] = None
    fireworks_api_key: Optional[str] = None
    fireworks_model: Optional[str] = None
    fireworks_async: bool = True
    fireworks_concurrency: int = 6
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = None
    groq_async: bool = True
    groq_concurrency: int = 6
    together_api_key: Optional[str] = None
    together_model: Optional[str] = None
    together_async: bool = True
    together_concurrency: int = 6
    ollama_model: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_async: bool = True
    ollama_concurrency: int = 6
    mistral_api_key: Optional[str] = None
    mistral_model: Optional[str] = None
    mistral_async: bool = True
    mistral_concurrency: int = 6
    nvidia_api_key: Optional[str] = None
    nvidia_model: Optional[str] = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    nvidia_async: bool = True
    nvidia_concurrency: int = 6
    rpm_limit: int = 60  # Requests per minute limit
    batch_translation: bool = False
    batch_validation_async: bool = False
    batch_validation_concurrency: int = 6
    chunk_size: int = 50
    sqlite_cache_extension: str = ".db"
    mod_name: Optional[str] = None
    use_mod_name: bool = False
    include_replace: bool = True
