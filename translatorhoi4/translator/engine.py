"""Translation engine and worker threads (ported from monolith)."""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from .backends import (
    TranslationBackend,
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
    _cleanup_llm_output,
    _strip_model_noise,
)
from .sqlite_cache import cache_factory
from .glossary import Glossary, _apply_replacements, _mask_glossary, _unmask_glossary
from .mask import mask_tokens, unmask_tokens, count_words_for_stats, _looks_like_http_error
from .prompts import batch_system_prompt, batch_wrap_with_markers, parse_batch_response
from .cost import cost_tracker, TokenUsage, TokenEstimator
from ..parsers.paradox_yaml import LOCALISATION_LINE_RE, HEADER_RE, SUPPORTED_LANG_HEADERS
from ..utils.fs import (
    collect_localisation_files,
    compute_output_path,
    _find_prev_localized_file,
    _build_prev_map,
    _combine_post_with_loc,
)


def _validate_translation(candidate: str, idx_tokens: list) -> bool:
    """Validate translation output after unmasking tokens."""
    if _looks_like_http_error(candidate):
        return False
    if '<<SEG' in candidate or '<<END' in candidate or '__TKN' in candidate:
        return False
    for tok in idx_tokens:
        if tok and tok not in candidate:
            return False
    return True


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
    rpm_limit: int = 60  # Requests per minute limit
    batch_translation: bool = False
    chunk_size: int = 50
    sqlite_cache_extension: str = ".db"
    mod_name: Optional[str] = None
    use_mod_name: bool = False


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
}


class TranslateWorker(QThread):
    progress = pyqtSignal(int, int)
    file_progress = pyqtSignal(str)
    file_inner_progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    stats = pyqtSignal(int, int, int)
    finished_ok = pyqtSignal()
    aborted = pyqtSignal(str)

    def __init__(self, cfg: JobConfig):
        super().__init__()
        self.cfg = cfg
        self._cancel = False
        self._words = 0
        self._keys = 0
        self._files_done = 0
        self._glossary = Glossary([], {})

        # Cost tracking
        self._session_tokens = TokenUsage()
        self._translated_keys = 0

        # Use cache factory to automatically choose SQLite for large projects
        cache_path = cfg.cache_path or os.path.join(cfg.out_dir or cfg.src_dir, ".hoi4loc_cache")
        self._cache = cache_factory.create_cache(path=cache_path, sqlite_extension=cfg.sqlite_cache_extension)

        if cfg.glossary_path and os.path.isfile(cfg.glossary_path):
            try:
                self._glossary = Glossary.load_csv(cfg.glossary_path)
            except Exception:
                pass

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            # Setup environment variables for backends that rely on them
            if self.cfg.model_key == "G4F: API (g4f.dev)":
                os.environ["G4F_MODEL"] = (self.cfg.g4f_model or "gpt-4o")
                os.environ["G4F_API_KEY"] = (self.cfg.g4f_api_key or "")
                os.environ["G4F_TEMP"] = str(self.cfg.temperature)
                os.environ["G4F_ASYNC"] = "1" if self.cfg.g4f_async else "0"
                os.environ["G4F_CONCURRENCY"] = str(self.cfg.g4f_concurrency)
            elif self.cfg.model_key == "IO: chat.completions":
                os.environ["IO_MODEL"] = (self.cfg.io_model or "meta-llama/Llama-3.3-70B-Instruct")
                os.environ["IO_API_KEY"] = (self.cfg.io_api_key or "")
                os.environ["IO_BASE_URL"] = (self.cfg.io_base_url or "https://api.intelligence.io.solutions/api/v1/")
                os.environ["IO_TEMP"] = str(self.cfg.temperature)
                os.environ["IO_ASYNC"] = "1" if self.cfg.io_async else "0"
                os.environ["IO_CONCURRENCY"] = str(self.cfg.io_concurrency)
            elif self.cfg.model_key == "OpenAI Compatible API":
                os.environ["OPENAI_MODEL"] = (self.cfg.openai_model or "gpt-4")
                os.environ["OPENAI_API_KEY"] = (self.cfg.openai_api_key or "")
                os.environ["OPENAI_BASE_URL"] = (self.cfg.openai_base_url or "https://api.openai.com/v1/")
                os.environ["OPENAI_TEMP"] = str(self.cfg.temperature)
                os.environ["OPENAI_ASYNC"] = "1" if self.cfg.openai_async else "0"
                os.environ["OPENAI_CONCURRENCY"] = str(self.cfg.openai_concurrency)
            elif self.cfg.model_key == "Anthropic: Claude":
                os.environ["ANTHROPIC_API_KEY"] = (self.cfg.anthropic_api_key or "")
                os.environ["ANTHROPIC_MODEL"] = (self.cfg.anthropic_model or "claude-sonnet-4-5-20250929")
                os.environ["ANTHROPIC_TEMP"] = str(self.cfg.temperature)
                os.environ["ANTHROPIC_ASYNC"] = "1" if self.cfg.anthropic_async else "0"
                os.environ["ANTHROPIC_CONCURRENCY"] = str(self.cfg.anthropic_concurrency)
            elif self.cfg.model_key == "Google: Gemini":
                os.environ["GEMINI_API_KEY"] = (self.cfg.gemini_api_key or "")
                os.environ["GEMINI_MODEL"] = (self.cfg.gemini_model or "gemini-2.5-flash")
                os.environ["GEMINI_TEMP"] = str(self.cfg.temperature)
                os.environ["GEMINI_ASYNC"] = "1" if self.cfg.gemini_async else "0"
                os.environ["GEMINI_CONCURRENCY"] = str(self.cfg.gemini_concurrency)
            elif self.cfg.model_key == "Yandex Translate":
                os.environ["YANDEX_TRANSLATE_API_KEY"] = (self.cfg.yandex_translate_api_key or "")
                os.environ["YANDEX_IAM_TOKEN"] = (self.cfg.yandex_iam_token or "")
                os.environ["YANDEX_FOLDER_ID"] = self.cfg.yandex_folder_id
            elif self.cfg.model_key == "Yandex Cloud":
                os.environ["YANDEX_CLOUD_API_KEY"] = (self.cfg.yandex_cloud_api_key or "")
                os.environ["YANDEX_CLOUD_MODEL"] = (self.cfg.yandex_cloud_model or "aliceai-llm/latest")
                os.environ["YANDEX_FOLDER_ID"] = self.cfg.yandex_folder_id
                os.environ["YANDEX_TEMP"] = str(self.cfg.temperature)
                os.environ["YANDEX_ASYNC"] = "1" if self.cfg.yandex_async else "0"
                os.environ["YANDEX_CONCURRENCY"] = str(self.cfg.yandex_concurrency)
            elif self.cfg.model_key == "DeepL API":
                os.environ["DEEPL_API_KEY"] = (self.cfg.deepl_api_key or "")
            elif self.cfg.model_key == "Fireworks.ai":
                os.environ["FIREWORKS_API_KEY"] = (self.cfg.fireworks_api_key or "")
                os.environ["FIREWORKS_MODEL"] = (self.cfg.fireworks_model or "accounts/fireworks/models/llama-v3p1-8b-instruct")
                os.environ["FIREWORKS_TEMP"] = str(self.cfg.temperature)
                os.environ["FIREWORKS_ASYNC"] = "1" if self.cfg.fireworks_async else "0"
                os.environ["FIREWORKS_CONCURRENCY"] = str(self.cfg.fireworks_concurrency)
            elif self.cfg.model_key == "Groq":
                os.environ["GROQ_API_KEY"] = (self.cfg.groq_api_key or "")
                os.environ["GROQ_MODEL"] = (self.cfg.groq_model or "openai/gpt-oss-20b")
                os.environ["GROQ_TEMP"] = str(self.cfg.temperature)
                os.environ["GROQ_ASYNC"] = "1" if self.cfg.groq_async else "0"
                os.environ["GROQ_CONCURRENCY"] = str(self.cfg.groq_concurrency)
            elif self.cfg.model_key == "Together.ai":
                os.environ["TOGETHER_API_KEY"] = (self.cfg.together_api_key or "")
                os.environ["TOGETHER_MODEL"] = (self.cfg.together_model or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
                os.environ["TOGETHER_TEMP"] = str(self.cfg.temperature)
                os.environ["TOGETHER_ASYNC"] = "1" if self.cfg.together_async else "0"
                os.environ["TOGETHER_CONCURRENCY"] = str(self.cfg.together_concurrency)
            elif self.cfg.model_key == "Ollama":
                os.environ["OLLAMA_MODEL"] = (self.cfg.ollama_model or "llama3.2")
                os.environ["OLLAMA_BASE_URL"] = self.cfg.ollama_base_url
                os.environ["OLLAMA_TEMP"] = str(self.cfg.temperature)
                os.environ["OLLAMA_ASYNC"] = "1" if self.cfg.ollama_async else "0"
                os.environ["OLLAMA_CONCURRENCY"] = str(self.cfg.ollama_concurrency)

            backend = MODEL_REGISTRY[self.cfg.model_key]()
            if hasattr(backend, 'temperature'):
                backend.temperature = self.cfg.temperature
            if hasattr(backend, 'rpm_limit'):
                backend.rpm_limit = self.cfg.rpm_limit
            backend.warmup()
        except Exception as e:
            self.aborted.emit(f"Failed to initialize backend: {e}\n{traceback.format_exc()}")
            return

        self._cache.load()
        files = collect_localisation_files(self.cfg.src_dir)
        total = len(files)
        fc = max(1, self.cfg.files_concurrency)
        idx_counter = 0
        stop_flag = False

        def process_one(path: str) -> Tuple[str, Optional[str]]:
            try:
                relname = os.path.relpath(path, self.cfg.src_dir)
                out_path = compute_output_path(path, self.cfg)
                if self.cfg.skip_existing and os.path.exists(out_path) and not self.cfg.in_place:
                    return (relname, None)
                
                # Add logging for directory creation
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Processing file: {path}")
                logger.info(f"Output path: {out_path}")
                logger.info(f"Output directory: {os.path.dirname(out_path)}")
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                logger.info(f"Directory created successfully: {os.path.dirname(out_path)}")
                
                with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
                    lines = f.readlines()
                prev_map: Dict[str, Tuple[str, str]] = {}
                if self.cfg.reuse_prev_loc and self.cfg.prev_loc_dir:
                    pf = _find_prev_localized_file(self.cfg.prev_loc_dir, relname, self.cfg.dst_lang)
                    if pf:
                        prev_map = _build_prev_map(pf)
                new_lines = self._process_file_lines(lines, backend, relname, prev_map)
                # Add logging for file creation
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Writing to file: {out_path}")
                
                with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
                    f.writelines(new_lines)
                logger.info(f"Successfully wrote to file: {out_path}")
                return (relname, None)
            except Exception as e:
                return (path, f"{e}\n{traceback.format_exc()}")

        try:
            with ThreadPoolExecutor(max_workers=fc, thread_name_prefix="file") as ex:
                futs = [ex.submit(process_one, p) for p in files]
                for fut in as_completed(futs):
                    if self._cancel:
                        stop_flag = True
                        break
                    rel, err = fut.result()
                    idx_counter += 1
                    if err is None:
                        out_path = compute_output_path(os.path.join(self.cfg.src_dir, rel), self.cfg)
                        if self.cfg.skip_existing and os.path.exists(out_path) and not self.cfg.in_place:
                            self.log.emit(f"[SKIP] {out_path}")
                        else:
                            self.log.emit(f"[OK] → {out_path}")
                    else:
                        self.log.emit(f"[ERR] {rel}: {err}")
                    self._files_done += 1
                    self.progress.emit(idx_counter, total)
        except Exception as e:
            self.aborted.emit(f"Executor failed: {e}\n{traceback.format_exc()}")
            return

        self._cache.save()

        # Record cost for the session
        if self._translated_keys > 0:
            self._record_session_cost()

        if stop_flag:
            self.aborted.emit("Cancelled by user")
        else:
            self.finished_ok.emit()

    def _process_file_lines(self, lines: List[str], backend: TranslationBackend, relname: str,
                             prev_map: Dict[str, Tuple[str, str]]) -> List[str]:
        # Use batch mode if enabled explicitly
        if self.cfg.batch_translation:
            return self._process_file_lines_batch(lines, backend, relname, prev_map)
            
        out: List[str] = []
        header_replaced = False
        key_count = sum(1 for ln in lines if LOCALISATION_LINE_RE.match(ln))
        done_keys = 0
        self.file_progress.emit(relname)
        self.file_inner_progress.emit(0, max(1, key_count))
        
        is_google = isinstance(backend, GoogleFreeBackend)
        # We removed G4F from here to force line-by-line smooth tracking
        # is_g4f = isinstance(backend, G4F_Backend) 
        
        batch_size = max(1, self.cfg.batch_size) if is_google else 1
        key_skip_re = re.compile(self.cfg.key_skip_regex) if self.cfg.key_skip_regex else None

        # Only use mini-batching loop for Google Translate (which is fast and needs it)
        if is_google:
            batch_buf: List[Tuple[str, Dict[str, str], List[str], Dict[str, str], Tuple[str, str, str, str, str]]] = []

            def flush_batch():
                nonlocal done_keys, out
                if not batch_buf:
                    return
                texts = [b[0] for b in batch_buf]
                try:
                    if getattr(backend, "supports_batch", False):
                        trans_list = backend.translate_many(texts, self.cfg.src_lang, self.cfg.dst_lang)
                    else:
                        trans_list = [backend.translate(t, self.cfg.src_lang, self.cfg.dst_lang) for t in texts]
                    for (masked, mapping, idx_tokens, glmap, groups), tr in zip(batch_buf, trans_list):
                        pre, key, version, old_text, post = groups
                        tr = _cleanup_llm_output(tr)
                        if self.cfg.strip_md:
                            tr = _strip_model_noise(tr)
                        candidate = unmask_tokens(tr.strip(), mapping, idx_tokens)
                        candidate = _unmask_glossary(candidate, glmap)
                        candidate = _apply_replacements(candidate, self._glossary)
                        if not _validate_translation(candidate, idx_tokens):
                            self.log.emit(f"[WARN] Bad output for key '{key}', keeping original.")
                            translated = old_text
                            cache_ok = False
                        else:
                            translated = candidate
                            cache_ok = True
                        post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
                        out.append(f"{pre}{key}:{version or 0} \"{translated}\"{post2}\n")
                        if cache_ok:
                            self._cache.set(masked, translated)
                        done_keys += 1
                        self._keys += 1
                        if self._keys % 20 == 0:
                            self.stats.emit(self._words, self._keys, self._files_done)
                        self.file_inner_progress.emit(done_keys, max(1, key_count))
                finally:
                    batch_buf.clear()

            for line in lines:
                if self._cancel:
                    break
                if not header_replaced:
                    m = HEADER_RE.match(line)
                    if m:
                        dst_header = SUPPORTED_LANG_HEADERS.get(self.cfg.dst_lang, f"l_{self.cfg.dst_lang}:")
                        out.append(dst_header + "\n")
                        header_replaced = True
                        continue
                m = LOCALISATION_LINE_RE.match(line)
                if not m:
                    out.append(line)
                    continue
                pre, key, version, text, post = m.groups()
                if key_skip_re and key_skip_re.search(key):
                    out.append(line)
                    continue
                if self.cfg.reuse_prev_loc and key in prev_map:
                    prev_text, prev_post = prev_map[key]
                    post2 = _combine_post_with_loc(prev_post or post or "", True)
                    out.append(f"{pre}{key}:{version or 0} \"{prev_text}\"{post2}\n")
                    done_keys += 1
                    self._keys += 1
                    self.file_inner_progress.emit(done_keys, max(1, key_count))
                    continue
                masked, mapping, idx_tokens = mask_tokens(text)
                masked, glmap = _mask_glossary(masked, self._glossary)
                self._words += count_words_for_stats(masked)
                cached = self._cache.get(masked)
                if cached is not None:
                    candidate = _unmask_glossary(cached, glmap)
                    candidate = _apply_replacements(candidate, self._glossary)
                    if not _validate_translation(candidate, idx_tokens):
                        candidate = text
                    post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
                    out.append(f"{pre}{key}:{version or 0} \"{candidate}\"{post2}\n")
                    done_keys += 1
                    self._keys += 1
                    self.file_inner_progress.emit(done_keys, max(1, key_count))
                else:
                    batch_buf.append((masked, mapping, idx_tokens, glmap, (pre, key, version, text, post)))
                    if len(batch_buf) >= batch_size:
                        flush_batch()
            flush_batch()
            self.stats.emit(self._words, self._keys, self._files_done)
            return out

        # Line-by-line translation for G4F/OpenAI/others (Smoother UI)
        for line in lines:
            if self._cancel:
                break
            if not header_replaced:
                m = HEADER_RE.match(line)
                if m:
                    dst_header = SUPPORTED_LANG_HEADERS.get(self.cfg.dst_lang, f"l_{self.cfg.dst_lang}:")
                    out.append(dst_header + "\n")
                    header_replaced = True
                    continue
            m = LOCALISATION_LINE_RE.match(line)
            if not m:
                out.append(line)
                continue
            pre, key, version, text, post = m.groups()
            if key_skip_re and key_skip_re.search(key):
                out.append(line)
                continue
            if self.cfg.reuse_prev_loc and key in prev_map:
                prev_text, prev_post = prev_map[key]
                post2 = _combine_post_with_loc(prev_post or post or "", True)
                out.append(f"{pre}{key}:{version or 0} \"{prev_text}\"{post2}\n")
                done_keys += 1
                self._keys += 1
                self.file_inner_progress.emit(done_keys, max(1, key_count))
                continue
            masked, mapping, idx_tokens = mask_tokens(text)
            masked, glmap = _mask_glossary(masked, self._glossary)
            self._words += count_words_for_stats(masked)
            cached = self._cache.get(masked)
            if cached is None:
                delay = 0.25
                translated = None
                for _ in range(4):
                    try:
                        tr = backend.translate(masked, self.cfg.src_lang, self.cfg.dst_lang)
                        tr = _cleanup_llm_output(tr)
                        if self.cfg.strip_md:
                            tr = _strip_model_noise(tr)
                        candidate = unmask_tokens(tr.strip(), mapping, idx_tokens)
                        candidate = _unmask_glossary(candidate, glmap)
                        candidate = _apply_replacements(candidate, self._glossary)
                        missing = 0
                        for tok in idx_tokens:
                            if tok and tok not in candidate:
                                missing += 1
                        bad = ('<<SEG' in candidate) or ('<<END' in candidate) or ('__TKN' in candidate) or _looks_like_http_error(candidate)
                        if missing == 0 and not bad:
                            translated = candidate
                            self._cache.set(masked, translated)
                            break
                    except Exception:
                        pass
                    time.sleep(delay)
                    delay *= 1.7
                if translated is None:
                    translated = text
            else:
                candidate = _unmask_glossary(cached, glmap)
                candidate = _apply_replacements(candidate, self._glossary)
                if ('<<SEG' in candidate) or ('<<END' in candidate) or ('__TKN' in candidate) or _looks_like_http_error(candidate):
                    candidate = text
                translated = candidate
            post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
            out.append(f"{pre}{key}:{version or 0} \"{translated}\"{post2}\n")
            done_keys += 1
            self._keys += 1
            if self._keys % 10 == 0:
                self.stats.emit(self._words, self._keys, self._files_done)
            self.file_inner_progress.emit(done_keys, max(1, key_count))
        self.stats.emit(self._words, self._keys, self._files_done)
        return out

    def _process_file_lines_batch(self, lines: List[str], backend: TranslationBackend, relname: str,
                                  prev_map: Dict[str, Tuple[str, str]]) -> List[str]:
        """Process file lines using batch translation mode."""
        processed_lines = lines[:]  # Copy all lines
        header_replaced = False
        key_count = sum(1 for ln in lines if LOCALISATION_LINE_RE.match(ln))
        done_keys = 0
        self.file_progress.emit(relname)
        self.file_inner_progress.emit(0, max(1, key_count))

        # Collect all translatable lines with their indices
        translatable_lines = []

        for idx, line in enumerate(lines):
            if not header_replaced:
                m = HEADER_RE.match(line)
                if m:
                    dst_header = SUPPORTED_LANG_HEADERS.get(self.cfg.dst_lang, f"l_{self.cfg.dst_lang}:")
                    processed_lines[idx] = dst_header + "\n"
                    header_replaced = True
                    continue

            m = LOCALISATION_LINE_RE.match(line)
            if not m:
                # Comments and empty lines remain as is
                continue

            pre, key, version, text, post = m.groups()

            # Skip if key matches skip regex
            key_skip_re = re.compile(self.cfg.key_skip_regex) if self.cfg.key_skip_regex else None
            if key_skip_re and key_skip_re.search(key):
                # Line remains unchanged
                continue

            # Skip if reusing previous localization
            if self.cfg.reuse_prev_loc and key in prev_map:
                prev_text, prev_post = prev_map[key]
                post2 = _combine_post_with_loc(prev_post or post or "", True)
                processed_lines[idx] = f"{pre}{key}:{version or 0} \"{prev_text}\"{post2}\n"
                done_keys += 1
                self._keys += 1
                self.file_inner_progress.emit(done_keys, max(1, key_count))
                continue

            # Prepare for translation
            masked, mapping, idx_tokens = mask_tokens(text)
            masked, glmap = _mask_glossary(masked, self._glossary)
            self._words += count_words_for_stats(masked)

            # Check cache first
            cached = self._cache.get(masked)
            if cached is not None:
                candidate = _unmask_glossary(cached, glmap)
                candidate = _apply_replacements(candidate, self._glossary)
                if not _validate_translation(candidate, idx_tokens):
                    candidate = text
                post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
                processed_lines[idx] = f"{pre}{key}:{version or 0} \"{candidate}\"{post2}\n"
                done_keys += 1
                self._keys += 1
                self.file_inner_progress.emit(done_keys, max(1, key_count))
                continue

            # Add to batch
            translatable_lines.append((idx, key, masked, mapping, idx_tokens, glmap, pre, version, text, post))
        
        # Process in chunks
        chunk_size = max(1, self.cfg.chunk_size)
        for i in range(0, len(translatable_lines), chunk_size):
            chunk = translatable_lines[i:i + chunk_size]
            if not chunk:
                continue

            # Create batch data
            batch_data = {}
            for idx, key, masked, _, _, _, _, _, _, _ in chunk:
                batch_data[key] = masked

            # Translate chunk
            try:
                batch_text = batch_wrap_with_markers(batch_data)

                # Use backend to translate
                response = backend.translate(batch_text, self.cfg.src_lang, self.cfg.dst_lang)
                translations = parse_batch_response(response)

                # Validate chunk
                if len(translations) != len(chunk):
                    self.log.emit(f"[WARN] Chunk validation failed for {relname}: expected {len(chunk)} translations, got {len(translations)}")
                    # Fall back to individual translation
                    for item in chunk:
                        idx, key, masked, mapping, idx_tokens, glmap, pre, version, text, post = item
                        try:
                            tr = backend.translate(masked, self.cfg.src_lang, self.cfg.dst_lang)
                            tr = _cleanup_llm_output(tr)
                            if self.cfg.strip_md:
                                tr = _strip_model_noise(tr)
                            candidate = unmask_tokens(tr.strip(), mapping, idx_tokens)
                            candidate = _unmask_glossary(candidate, glmap)
                            candidate = _apply_replacements(candidate, self._glossary)
                            if not _validate_translation(candidate, idx_tokens):
                                candidate = text
                            post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
                            processed_lines[idx] = f"{pre}{key}:{version or 0} \"{candidate}\"{post2}\n"
                            self._cache.set(masked, candidate)
                            done_keys += 1
                            self._keys += 1
                            self.file_inner_progress.emit(done_keys, max(1, key_count))
                        except Exception as e:
                            self.log.emit(f"[ERR] Individual translation failed for {key}: {e}")
                            processed_lines[idx] = f"{pre}{key}:{version or 0} \"{text}\"{post}\n"
                            done_keys += 1
                            self.file_inner_progress.emit(done_keys, max(1, key_count))
                else:
                    # Process successful batch
                    for item in chunk:
                        idx, key, masked, mapping, idx_tokens, glmap, pre, version, text, post = item
                        if key in translations:
                            tr = translations[key]
                            tr = _cleanup_llm_output(tr)
                            if self.cfg.strip_md:
                                tr = _strip_model_noise(tr)
                            candidate = unmask_tokens(tr.strip(), mapping, idx_tokens)
                            candidate = _unmask_glossary(candidate, glmap)
                            candidate = _apply_replacements(candidate, self._glossary)
                            if not _validate_translation(candidate, idx_tokens):
                                candidate = text
                            post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
                            processed_lines[idx] = f"{pre}{key}:{version or 0} \"{candidate}\"{post2}\n"
                            self._cache.set(masked, candidate)
                        else:
                            # Key not found in response, keep original
                            processed_lines[idx] = f"{pre}{key}:{version or 0} \"{text}\"{post}\n"

                        done_keys += 1
                        self._keys += 1
                        if self._keys % 20 == 0:
                            self.stats.emit(self._words, self._keys, self._files_done)
                        self.file_inner_progress.emit(done_keys, max(1, key_count))

            except Exception as e:
                self.log.emit(f"[ERR] Batch translation failed for {relname}: {e}")
                # Fall back to individual translation
                for item in chunk:
                    idx, key, masked, mapping, idx_tokens, glmap, pre, version, text, post = item
                    try:
                        tr = backend.translate(masked, self.cfg.src_lang, self.cfg.dst_lang)
                        tr = _cleanup_llm_output(tr)
                        if self.cfg.strip_md:
                            tr = _strip_model_noise(tr)
                        candidate = unmask_tokens(tr.strip(), mapping, idx_tokens)
                        candidate = _unmask_glossary(candidate, glmap)
                        candidate = _apply_replacements(candidate, self._glossary)
                        if not _validate_translation(candidate, idx_tokens):
                            candidate = text
                        post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
                        processed_lines[idx] = f"{pre}{key}:{version or 0} \"{candidate}\"{post2}\n"
                        self._cache.set(masked, candidate)
                        done_keys += 1
                        self._keys += 1
                        self.file_inner_progress.emit(done_keys, max(1, key_count))
                    except Exception as e:
                        self.log.emit(f"[ERR] Individual translation failed for {key}: {e}")
                        processed_lines[idx] = f"{pre}{key}:{version or 0} \"{text}\"{post}\n"
                        done_keys += 1
                        self.file_inner_progress.emit(done_keys, max(1, key_count))

        self.stats.emit(self._words, self._keys, self._files_done)
        return processed_lines


class RetranslateWorker(QThread):
    progress = pyqtSignal(int, int)  # current, total
    translation_done = pyqtSignal(list)  # list of {'key': str, 'translation': str, 'row': int}
    log = pyqtSignal(str)

    def __init__(self, cfg: JobConfig, items: List[Dict]):
        super().__init__()
        self.cfg = cfg
        self.items = items
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            backend = MODEL_REGISTRY[self.cfg.model_key]()
            if hasattr(backend, 'temperature'):
                backend.temperature = self.cfg.temperature
            if hasattr(backend, 'rpm_limit'):
                backend.rpm_limit = self.cfg.rpm_limit
            backend.warmup()

            # Setup environment variables
            if self.cfg.model_key == "G4F: API (g4f.dev)":
                os.environ["G4F_MODEL"] = (self.cfg.g4f_model or "gpt-4o")
                os.environ["G4F_API_KEY"] = (self.cfg.g4f_api_key or "")
                os.environ["G4F_TEMP"] = str(self.cfg.temperature)
                os.environ["G4F_ASYNC"] = "1" if self.cfg.g4f_async else "0"
                os.environ["G4F_CONCURRENCY"] = str(self.cfg.g4f_concurrency)
            elif self.cfg.model_key == "IO: chat.completions":
                os.environ["IO_MODEL"] = (self.cfg.io_model or "meta-llama/Llama-3.3-70B-Instruct")
                os.environ["IO_API_KEY"] = (self.cfg.io_api_key or "")
                os.environ["IO_BASE_URL"] = (self.cfg.io_base_url or "https://api.intelligence.io.solutions/api/v1/")
                os.environ["IO_TEMP"] = str(self.cfg.temperature)
                os.environ["IO_ASYNC"] = "1" if self.cfg.io_async else "0"
                os.environ["IO_CONCURRENCY"] = str(self.cfg.io_concurrency)
            elif self.cfg.model_key == "OpenAI Compatible API":
                os.environ["OPENAI_MODEL"] = (self.cfg.openai_model or "gpt-4")
                os.environ["OPENAI_API_KEY"] = (self.cfg.openai_api_key or "")
                os.environ["OPENAI_BASE_URL"] = (self.cfg.openai_base_url or "https://api.openai.com/v1/")
                os.environ["OPENAI_TEMP"] = str(self.cfg.temperature)
                os.environ["OPENAI_ASYNC"] = "1" if self.cfg.openai_async else "0"
                os.environ["OPENAI_CONCURRENCY"] = str(self.cfg.openai_concurrency)
            elif self.cfg.model_key == "Anthropic: Claude":
                os.environ["ANTHROPIC_API_KEY"] = (self.cfg.anthropic_api_key or "")
                os.environ["ANTHROPIC_MODEL"] = (self.cfg.anthropic_model or "claude-sonnet-4-5-20250929")
                os.environ["ANTHROPIC_TEMP"] = str(self.cfg.temperature)
                os.environ["ANTHROPIC_ASYNC"] = "1" if self.cfg.anthropic_async else "0"
                os.environ["ANTHROPIC_CONCURRENCY"] = str(self.cfg.anthropic_concurrency)
            elif self.cfg.model_key == "Google: Gemini":
                os.environ["GEMINI_API_KEY"] = (self.cfg.gemini_api_key or "")
                os.environ["GEMINI_MODEL"] = (self.cfg.gemini_model or "gemini-2.5-flash")
                os.environ["GEMINI_TEMP"] = str(self.cfg.temperature)
                os.environ["GEMINI_ASYNC"] = "1" if self.cfg.gemini_async else "0"
                os.environ["GEMINI_CONCURRENCY"] = str(self.cfg.gemini_concurrency)
            elif self.cfg.model_key == "Yandex Translate":
                os.environ["YANDEX_TRANSLATE_API_KEY"] = (self.cfg.yandex_translate_api_key or "")
                os.environ["YANDEX_IAM_TOKEN"] = (self.cfg.yandex_iam_token or "")
                os.environ["YANDEX_FOLDER_ID"] = self.cfg.yandex_folder_id
            elif self.cfg.model_key == "Yandex Cloud":
                os.environ["YANDEX_CLOUD_API_KEY"] = (self.cfg.yandex_cloud_api_key or "")
                os.environ["YANDEX_CLOUD_MODEL"] = (self.cfg.yandex_cloud_model or "aliceai-llm/latest")
                os.environ["YANDEX_FOLDER_ID"] = self.cfg.yandex_folder_id
                os.environ["YANDEX_TEMP"] = str(self.cfg.temperature)
                os.environ["YANDEX_ASYNC"] = "1" if self.cfg.yandex_async else "0"
                os.environ["YANDEX_CONCURRENCY"] = str(self.cfg.yandex_concurrency)
            elif self.cfg.model_key == "DeepL API":
                os.environ["DEEPL_API_KEY"] = (self.cfg.deepl_api_key or "")
            elif self.cfg.model_key == "Fireworks.ai":
                os.environ["FIREWORKS_API_KEY"] = (self.cfg.fireworks_api_key or "")
                os.environ["FIREWORKS_MODEL"] = (self.cfg.fireworks_model or "accounts/fireworks/models/llama-v3p1-8b-instruct")
                os.environ["FIREWORKS_TEMP"] = str(self.cfg.temperature)
                os.environ["FIREWORKS_ASYNC"] = "1" if self.cfg.fireworks_async else "0"
                os.environ["FIREWORKS_CONCURRENCY"] = str(self.cfg.fireworks_concurrency)
            elif self.cfg.model_key == "Groq":
                os.environ["GROQ_API_KEY"] = (self.cfg.groq_api_key or "")
                os.environ["GROQ_MODEL"] = (self.cfg.groq_model or "openai/gpt-oss-20b")
                os.environ["GROQ_TEMP"] = str(self.cfg.temperature)
                os.environ["GROQ_ASYNC"] = "1" if self.cfg.groq_async else "0"
                os.environ["GROQ_CONCURRENCY"] = str(self.cfg.groq_concurrency)
            elif self.cfg.model_key == "Together.ai":
                os.environ["TOGETHER_API_KEY"] = (self.cfg.together_api_key or "")
                os.environ["TOGETHER_MODEL"] = (self.cfg.together_model or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
                os.environ["TOGETHER_TEMP"] = str(self.cfg.temperature)
                os.environ["TOGETHER_ASYNC"] = "1" if self.cfg.together_async else "0"
                os.environ["TOGETHER_CONCURRENCY"] = str(self.cfg.together_concurrency)
            elif self.cfg.model_key == "Ollama":
                os.environ["OLLAMA_MODEL"] = (self.cfg.ollama_model or "llama3.2")
                os.environ["OLLAMA_BASE_URL"] = self.cfg.ollama_base_url
                os.environ["OLLAMA_TEMP"] = str(self.cfg.temperature)
                os.environ["OLLAMA_ASYNC"] = "1" if self.cfg.ollama_async else "0"
                os.environ["OLLAMA_CONCURRENCY"] = str(self.cfg.ollama_concurrency)

            # Load glossary
            glossary = Glossary([], {})
            if self.cfg.glossary_path and os.path.isfile(self.cfg.glossary_path):
                try:
                    glossary = Glossary.load_csv(self.cfg.glossary_path)
                except Exception:
                    pass

            # Load cache
            cache_path = self.cfg.cache_path or os.path.join(self.cfg.out_dir or self.cfg.src_dir, ".hoi4loc_cache")
            cache = cache_factory.create_cache(path=cache_path, sqlite_extension=self.cfg.sqlite_cache_extension)
            cache.load()

            results = []
            total = len(self.items)
            for i, item in enumerate(self.items):
                if self._cancel:
                    break
                key = item['key']
                original = item['original']
                row = item['row']

                try:
                    masked, mapping, idx_tokens = mask_tokens(original)
                    masked, glmap = _mask_glossary(masked, glossary)

                    cached = cache.get(masked)
                    if cached is None:
                        tr = backend.translate(masked, self.cfg.src_lang, self.cfg.dst_lang)
                        tr = _cleanup_llm_output(tr)
                        if self.cfg.strip_md:
                            tr = _strip_model_noise(tr)
                        candidate = unmask_tokens(tr.strip(), mapping, idx_tokens)
                        candidate = _unmask_glossary(candidate, glmap)
                        candidate = _apply_replacements(candidate, glossary)
                        if not _validate_translation(candidate, idx_tokens):
                            self.log.emit(f"[WARN] Bad output for key '{key}', keeping original.")
                            candidate = original
                        cache.set(masked, candidate)
                    else:
                        candidate = _unmask_glossary(cached, glmap)
                        candidate = _apply_replacements(candidate, glossary)
                        if not _validate_translation(candidate, idx_tokens):
                            candidate = original

                    results.append({'key': key, 'translation': candidate, 'row': row})
                except Exception as e:
                    self.log.emit(f"[ERR] Failed to translate '{key}': {e}")
                    results.append({'key': key, 'translation': original, 'row': row})

                self.progress.emit(i + 1, total)

            cache.save()
            self.translation_done.emit(results)

        except Exception as e:
            self.log.emit(f"[ERR] Retranslate failed: {e}")


class TestModelWorker(QThread):
    ok = pyqtSignal(str)
    fail = pyqtSignal(str)

    def __init__(self, model_key: str, src_lang: str, dst_lang: str, temperature: float,
                 strip_md: bool, glossary_path: Optional[str],
                 g4f_model: Optional[str], g4f_api_key: Optional[str],
                 g4f_async: bool, g4f_concurrency: int,
                 io_model: Optional[str], io_api_key: Optional[str], io_base_url: Optional[str],
                 io_async: bool, io_concurrency: int,
                 openai_api_key: Optional[str], openai_model: Optional[str], openai_base_url: Optional[str],
                 openai_async: bool, openai_concurrency: int,
                 anthropic_api_key: Optional[str], anthropic_model: Optional[str], anthropic_async: bool, anthropic_concurrency: int,
                 gemini_api_key: Optional[str], gemini_model: Optional[str], gemini_async: bool, gemini_concurrency: int,
                 yandex_translate_api_key: Optional[str], yandex_iam_token: Optional[str], yandex_folder_id: str,
                 yandex_cloud_api_key: Optional[str], yandex_cloud_model: Optional[str], yandex_async: bool, yandex_concurrency: int,
                 deepl_api_key: Optional[str],
                 fireworks_api_key: Optional[str], fireworks_model: Optional[str], fireworks_async: bool, fireworks_concurrency: int,
                 groq_api_key: Optional[str], groq_model: Optional[str], groq_async: bool, groq_concurrency: int,
                 together_api_key: Optional[str], together_model: Optional[str], together_async: bool, together_concurrency: int,
                 ollama_model: Optional[str], ollama_base_url: str, ollama_async: bool, ollama_concurrency: int,
                 sqlite_cache_extension: str = ".db"):
        super().__init__()
        self.model_key = model_key
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.temperature = temperature
        self.strip_md = strip_md
        self.glossary_path = glossary_path
        self.g4f_model = g4f_model
        self.g4f_api_key = g4f_api_key
        self.g4f_async = g4f_async
        self.g4f_concurrency = g4f_concurrency
        self.io_model = io_model
        self.io_api_key = io_api_key
        self.io_base_url = io_base_url
        self.io_async = io_async
        self.io_concurrency = io_concurrency
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url
        self.openai_async = openai_async
        self.openai_concurrency = openai_concurrency
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_model = anthropic_model
        self.anthropic_async = anthropic_async
        self.anthropic_concurrency = anthropic_concurrency
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.gemini_async = gemini_async
        self.gemini_concurrency = gemini_concurrency
        self.yandex_translate_api_key = yandex_translate_api_key
        self.yandex_iam_token = yandex_iam_token
        self.yandex_folder_id = yandex_folder_id
        self.yandex_cloud_api_key = yandex_cloud_api_key
        self.yandex_cloud_model = yandex_cloud_model
        self.yandex_async = yandex_async
        self.yandex_concurrency = yandex_concurrency
        self.deepl_api_key = deepl_api_key
        self.fireworks_api_key = fireworks_api_key
        self.fireworks_model = fireworks_model
        self.fireworks_async = fireworks_async
        self.fireworks_concurrency = fireworks_concurrency
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.groq_async = groq_async
        self.groq_concurrency = groq_concurrency
        self.together_api_key = together_api_key
        self.together_model = together_model
        self.together_async = together_async
        self.together_concurrency = together_concurrency
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.ollama_async = ollama_async
        self.ollama_concurrency = ollama_concurrency

    def run(self):
        try:
            gl = Glossary([], {})
            if self.glossary_path and os.path.isfile(self.glossary_path):
                gl = Glossary.load_csv(self.glossary_path)
            if self.model_key == "G4F: API (g4f.dev)":
                os.environ["G4F_MODEL"] = (self.g4f_model or "gpt-4o")
                os.environ["G4F_API_KEY"] = (self.g4f_api_key or "")
                os.environ["G4F_TEMP"] = str(self.temperature)
                os.environ["G4F_ASYNC"] = "1" if self.g4f_async else "0"
                os.environ["G4F_CONCURRENCY"] = str(self.g4f_concurrency)
            elif self.model_key == "IO: chat.completions":
                os.environ["IO_MODEL"] = (self.io_model or "meta-llama/Llama-3.3-70B-Instruct")
                os.environ["IO_API_KEY"] = (self.io_api_key or "")
                os.environ["IO_BASE_URL"] = (self.io_base_url or "https://api.intelligence.io.solutions/api/v1/")
                os.environ["IO_TEMP"] = str(self.temperature)
                os.environ["IO_ASYNC"] = "1" if self.io_async else "0"
                os.environ["IO_CONCURRENCY"] = str(self.io_concurrency)

            elif self.model_key == "OpenAI Compatible API":
                os.environ["OPENAI_MODEL"] = (self.openai_model or "gpt-4")
                os.environ["OPENAI_API_KEY"] = (self.openai_api_key or "")
                os.environ["OPENAI_BASE_URL"] = (self.openai_base_url or "https://api.openai.com/v1/")
                os.environ["OPENAI_TEMP"] = str(self.temperature)
                os.environ["OPENAI_ASYNC"] = "1" if self.openai_async else "0"
                os.environ["OPENAI_CONCURRENCY"] = str(self.openai_concurrency)
            elif self.model_key == "Anthropic: Claude":
                os.environ["ANTHROPIC_API_KEY"] = (self.anthropic_api_key or "")
                os.environ["ANTHROPIC_MODEL"] = (self.anthropic_model or "claude-sonnet-4-5-20250929")
                os.environ["ANTHROPIC_TEMP"] = str(self.temperature)
                os.environ["ANTHROPIC_ASYNC"] = "1" if self.anthropic_async else "0"
                os.environ["ANTHROPIC_CONCURRENCY"] = str(self.anthropic_concurrency)
            elif self.model_key == "Google: Gemini":
                os.environ["GEMINI_API_KEY"] = (self.gemini_api_key or "")
                os.environ["GEMINI_MODEL"] = (self.gemini_model or "gemini-2.5-flash")
                os.environ["GEMINI_TEMP"] = str(self.temperature)
                os.environ["GEMINI_ASYNC"] = "1" if self.gemini_async else "0"
                os.environ["GEMINI_CONCURRENCY"] = str(self.gemini_concurrency)
            elif self.model_key == "Yandex Translate":
                os.environ["YANDEX_TRANSLATE_API_KEY"] = (self.yandex_translate_api_key or "")
                os.environ["YANDEX_IAM_TOKEN"] = (self.yandex_iam_token or "")
                os.environ["YANDEX_FOLDER_ID"] = self.yandex_folder_id
            elif self.model_key == "Yandex Cloud":
                os.environ["YANDEX_CLOUD_API_KEY"] = (self.yandex_cloud_api_key or "")
                os.environ["YANDEX_CLOUD_MODEL"] = (self.yandex_cloud_model or "aliceai-llm/latest")
                os.environ["YANDEX_FOLDER_ID"] = self.yandex_folder_id
                os.environ["YANDEX_TEMP"] = str(self.temperature)
                os.environ["YANDEX_ASYNC"] = "1" if self.yandex_async else "0"
                os.environ["YANDEX_CONCURRENCY"] = str(self.yandex_concurrency)
            elif self.model_key == "DeepL API":
                os.environ["DEEPL_API_KEY"] = (self.deepl_api_key or "")
            elif self.model_key == "Fireworks.ai":
                os.environ["FIREWORKS_API_KEY"] = (self.fireworks_api_key or "")
                os.environ["FIREWORKS_MODEL"] = (self.fireworks_model or "accounts/fireworks/models/llama-v3p1-8b-instruct")
                os.environ["FIREWORKS_TEMP"] = str(self.temperature)
                os.environ["FIREWORKS_ASYNC"] = "1" if self.fireworks_async else "0"
                os.environ["FIREWORKS_CONCURRENCY"] = str(self.fireworks_concurrency)
            elif self.model_key == "Groq":
                os.environ["GROQ_API_KEY"] = (self.groq_api_key or "")
                os.environ["GROQ_MODEL"] = (self.groq_model or "openai/gpt-oss-20b")
                os.environ["GROQ_TEMP"] = str(self.temperature)
                os.environ["GROQ_ASYNC"] = "1" if self.groq_async else "0"
                os.environ["GROQ_CONCURRENCY"] = str(self.groq_concurrency)
            elif self.model_key == "Together.ai":
                os.environ["TOGETHER_API_KEY"] = (self.together_api_key or "")
                os.environ["TOGETHER_MODEL"] = (self.together_model or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
                os.environ["TOGETHER_TEMP"] = str(self.temperature)
                os.environ["TOGETHER_ASYNC"] = "1" if self.together_async else "0"
                os.environ["TOGETHER_CONCURRENCY"] = str(self.together_concurrency)
            elif self.model_key == "Ollama":
                os.environ["OLLAMA_MODEL"] = (self.ollama_model or "llama3.2")
                os.environ["OLLAMA_BASE_URL"] = self.ollama_base_url
                os.environ["OLLAMA_TEMP"] = str(self.temperature)
                os.environ["OLLAMA_ASYNC"] = "1" if self.ollama_async else "0"
                os.environ["OLLAMA_CONCURRENCY"] = str(self.ollama_concurrency)

            backend = MODEL_REGISTRY[self.model_key]()
            if hasattr(backend, 'temperature'):
                backend.temperature = self.temperature
            backend.warmup()
            masked, mapping, idx = mask_tokens("Hello world and $COUNTRY$ [new_controller.GetAdjective]! Use %d and \\n.")
            masked, glmap = _mask_glossary(masked, gl)
            out = backend.translate(masked, self.src_lang, self.dst_lang)
            out = _cleanup_llm_output(out)
            if self.strip_md:
                out = _strip_model_noise(out)
            out = unmask_tokens(out, mapping, idx)
            out = _unmask_glossary(out, glmap)
            out = _apply_replacements(out, gl)
            self.ok.emit(out)
        except Exception as e:
            self.fail.emit(f"{e}\n{traceback.format_exc()}")

    def _record_session_cost(self):
        """Record the cost for this translation session."""
        try:
            # Map model key to provider
            provider_map = {
                "G4F: API (g4f.dev)": "g4f",
                "IO: chat.completions": "io",
                "OpenAI Compatible API": "openai",
                "Anthropic: Claude": "anthropic",
                "Google: Gemini": "gemini",
                "Google (free unofficial)": "google_free",
                "Yandex Translate": "yandex_translate",
                "Yandex Cloud": "yandex_cloud",
                "DeepL API": "deepl",
                "Fireworks.ai": "fireworks",
                "Groq": "groq",
                "Together.ai": "together",
                "Ollama": "ollama",
            }

            provider = provider_map.get(self.cfg.model_key, "unknown")

            # Estimate completion tokens (translations are typically longer)
            estimated_completion_tokens = TokenEstimator.estimate_completion_tokens(
                "",  # We don't have the original text easily
                language=self.cfg.dst_lang
            ) * self._translated_keys

            # For simplicity, assume prompt tokens are about 1/3 of completion
            estimated_prompt_tokens = estimated_completion_tokens // 3

            usage = TokenUsage(
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens
            )

            cost_tracker.record_usage(
                provider=provider,
                model=self.cfg.model_key,
                usage=usage,
                source_lang=self.cfg.src_lang,
                target_lang=self.cfg.dst_lang,
                entry_count=self._translated_keys,
            )

        except Exception:
            # Don't fail the translation if cost tracking fails
            pass