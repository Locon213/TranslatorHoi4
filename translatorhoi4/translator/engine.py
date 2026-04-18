"""Translation engine and worker threads.

This module re-exports JobConfig and MODEL_REGISTRY from their dedicated
modules (config.py and registry.py) and defines the worker QThreads.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Tuple
import threading

from PySide6.QtCore import QThread, Signal

from .backends import (
    TranslationBackend,
    GoogleFreeBackend,
    _cleanup_llm_output,
    _strip_model_noise,
)
from .config import JobConfig  # noqa: F401 — re-exported
from .registry import MODEL_REGISTRY  # noqa: F401 — re-exported
from .sqlite_cache import cache_factory
from .glossary import Glossary, _apply_replacements, _mask_glossary, _unmask_glossary
from .mask import mask_tokens, unmask_tokens, count_words_for_stats, _looks_like_http_error
from .prompts import batch_wrap_with_markers, parse_batch_response
from .cost import cost_tracker, TokenUsage, TokenEstimator
from ..parsers.paradox_yaml import LOCALISATION_LINE_RE, HEADER_RE, SUPPORTED_LANG_HEADERS
from ..utils.fs import (
    collect_localisation_files,
    collect_localisation_files_by_lang,
    collect_localisation_files_parallel,
    collect_source_language_files,
    compute_output_path,
    _find_prev_localized_file,
    _build_prev_map,
    _combine_post_with_loc,
)


def _validate_translation(candidate: str, idx_tokens: list) -> bool:
    """Validate translation output after unmasking tokens.
    
    Checks:
    - No HTTP error patterns
    - No segment markers left in output
    - No unmasked tokens (they should have been replaced)
    - Not empty or whitespace only
    """
    # Check for empty/whitespace
    if not candidate or not candidate.strip():
        return False
    
    # Check for error patterns
    if _looks_like_http_error(candidate):
        return False
    
    # Check for leftover markers
    if '<<SEG' in candidate or '<<END' in candidate or '__TKN' in candidate:
        return False
    
    # Check that all index tokens have been replaced (they should NOT be in candidate)
    for tok in idx_tokens:
        if tok and tok in candidate:
            # Token was not replaced - this is an error
            return False
    
    return True


class TranslateWorker(QThread):
    progress = Signal(int, int)
    file_progress = Signal(str)
    file_inner_progress = Signal(int, int)
    log = Signal(str)
    stats = Signal(int, int, int)
    finished_ok = Signal()
    aborted = Signal(str)

    def __init__(self, cfg: JobConfig):
        super().__init__()
        self.cfg = cfg
        self._cancel = False
        self._force_cancel_flag = False  # Flag for immediate hard stop
        self._paused_event = threading.Event()
        self._paused_event.set()  # Start in "not paused" state
        self._words = 0
        self._keys = 0
        self._files_done = 0
        self._glossary = Glossary([], {})

        # Cost tracking
        self._session_tokens = TokenUsage()
        self._translated_keys = 0
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "provider_calls": 0,
            "provider_items": 0,
            "provider_time_s": 0.0,
            "structured_batch_attempts": 0,
            "structured_batch_failures": 0,
            "structured_batch_fallbacks": 0,
            "translate_many_batches": 0,
        }

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
        self._paused_event.set()  # Resume so thread can exit

    def pause(self):
        """Pause the translation process."""
        self._paused_event.clear()

    def resume(self):
        """Resume the translation process."""
        self._paused_event.set()

    def is_paused(self) -> bool:
        return not self._paused_event.is_set()

    def force_cancel(self):
        """Request a fast cooperative stop without terminating the thread."""
        self._cancel = True
        self._force_cancel_flag = True
        self._paused_event.set()
        try:
            self._cache.save()
        except Exception:
            pass

    def _check_paused(self):
        """Block while paused. Returns True if should continue, False if cancelled."""
        while not self._paused_event.wait(timeout=0.1):
            if self._cancel_requested():
                return False
        return not self._cancel_requested()

    def _check_force_cancel(self):
        """Check if force cancel was requested - returns True if should stop immediately."""
        return self._force_cancel_flag

    def _cancel_requested(self) -> bool:
        return self._cancel or self._force_cancel_flag or self.isInterruptionRequested()

    def _format_output_line(self, pre: str, key: str, version: str, text: str, post: str) -> str:
        post2 = _combine_post_with_loc(post, self.cfg.mark_loc_flag)
        return f"{pre}{key}:{version or 0} \"{text}\"{post2}\n"

    def _bulk_cache_get(self, keys: List[str]) -> Dict[str, str]:
        if not keys:
            return {}
        if hasattr(self._cache, "get_many"):
            try:
                return self._cache.get_many(keys)
            except Exception:
                pass
        result = {}
        for key in keys:
            value = self._cache.get(key)
            if value is not None:
                result[key] = value
        return result

    def _bulk_cache_set(self, entries: Dict[str, str]) -> None:
        if not entries:
            return
        if hasattr(self._cache, "set_many"):
            try:
                self._cache.set_many(entries)
                return
            except Exception:
                pass
        for key, value in entries.items():
            self._cache.set(key, value)

    def _finalize_candidate(self, item: Dict[str, object], translated_text: str) -> Tuple[str, bool]:
        candidate = unmask_tokens(translated_text.strip(), item["mapping"], item["idx_tokens"])
        candidate = _unmask_glossary(candidate, item["glmap"])
        candidate = _apply_replacements(candidate, self._glossary)
        if not _validate_translation(candidate, item["idx_tokens"]):
            return item["text"], False
        return candidate, True

    def _timed_translate_many(self, backend: TranslationBackend, texts: List[str]) -> List[str]:
        if not texts:
            return []
        self._metrics["provider_calls"] += 1
        self._metrics["provider_items"] += len(texts)
        if len(texts) > 1:
            self._metrics["translate_many_batches"] += 1
        started = time.perf_counter()
        try:
            if len(texts) == 1 or not getattr(backend, "supports_batch", False):
                return [backend.translate_one(texts[0], self.cfg.src_lang, self.cfg.dst_lang)]

            results = backend.translate_many(texts, self.cfg.src_lang, self.cfg.dst_lang)
            if len(results) != len(texts):
                raise ValueError(f"translate_many returned {len(results)} results for {len(texts)} texts")
            return results
        except Exception:
            return [backend.translate_one(text, self.cfg.src_lang, self.cfg.dst_lang) for text in texts]
        finally:
            self._metrics["provider_time_s"] += (time.perf_counter() - started)

    def _timed_translate_structured_batch(self, backend: TranslationBackend, payload: str, item_count: int) -> str:
        self._metrics["structured_batch_attempts"] += 1
        self._metrics["provider_calls"] += 1
        self._metrics["provider_items"] += item_count
        started = time.perf_counter()
        try:
            return backend.translate_structured_batch(payload, self.cfg.src_lang, self.cfg.dst_lang)
        finally:
            self._metrics["provider_time_s"] += (time.perf_counter() - started)

    def _log_performance_summary(self) -> None:
        total_cache = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        cache_hit_rate = (self._metrics["cache_hits"] / total_cache * 100.0) if total_cache else 0.0
        avg_provider_ms = (
            self._metrics["provider_time_s"] / self._metrics["provider_calls"] * 1000.0
            if self._metrics["provider_calls"] else 0.0
        )
        self.log.emit(
            "[PERF] "
            f"cache_hit_rate={cache_hit_rate:.1f}% "
            f"hits={self._metrics['cache_hits']} misses={self._metrics['cache_misses']} "
            f"provider_calls={self._metrics['provider_calls']} items={self._metrics['provider_items']} "
            f"avg_provider_ms={avg_provider_ms:.1f} "
            f"structured_batch_failures={self._metrics['structured_batch_failures']} "
            f"structured_batch_fallbacks={self._metrics['structured_batch_fallbacks']} "
            f"translate_many_batches={self._metrics['translate_many_batches']}"
        )

    def run(self):
        try:
            # Setup game context for translation
            os.environ["GAME_ID"] = (self.cfg.game_id or "hoi4")
            if self.cfg.mod_theme:
                os.environ["MOD_THEME"] = self.cfg.mod_theme
            else:
                os.environ["MOD_THEME"] = ""
            
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
            elif self.cfg.model_key == "Mistral AI":
                os.environ["MISTRAL_API_KEY"] = (self.cfg.mistral_api_key or "")
                os.environ["MISTRAL_MODEL"] = (self.cfg.mistral_model or "mistral-small-latest")
                os.environ["MISTRAL_TEMP"] = str(self.cfg.temperature)
                os.environ["MISTRAL_ASYNC"] = "1" if self.cfg.mistral_async else "0"
                os.environ["MISTRAL_CONCURRENCY"] = str(self.cfg.mistral_concurrency)
            elif self.cfg.model_key == "Nvidia NIM":
                os.environ["NVIDIA_API_KEY"] = (self.cfg.nvidia_api_key or "")
                os.environ["NVIDIA_MODEL"] = (self.cfg.nvidia_model or "moonshotai/kimi-k2.5")
                os.environ["NVIDIA_BASE_URL"] = (self.cfg.nvidia_base_url or "https://integrate.api.nvidia.com/v1/chat/completions")
                os.environ["NVIDIA_TEMP"] = str(self.cfg.temperature)
                os.environ["NVIDIA_ASYNC"] = "1" if self.cfg.nvidia_async else "0"
                os.environ["NVIDIA_CONCURRENCY"] = str(self.cfg.nvidia_concurrency)

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
        
        # Use selective file collection based on source language and replace folder settings
        # Use parallel collection for large mods
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            use_parallel = available_memory > 2 * 1024 * 1024 * 1024  # 2GB available
        except:
            use_parallel = False
        
        if self.cfg.src_lang:
            files = collect_source_language_files(self.cfg.src_dir, self.cfg.src_lang)
            if not files:
                # Fallback to all files if no source language files found
                if use_parallel and len(os.listdir(self.cfg.src_dir)) > 50:
                    self.log.emit("Using parallel file collection for large mod...")
                    if self.cfg.include_replace:
                        files = collect_localisation_files_parallel(self.cfg.src_dir)
                    else:
                        files = collect_localisation_files_by_lang(self.cfg.src_dir, [self.cfg.src_lang], include_replace=False)
                else:
                    if self.cfg.include_replace:
                        files = collect_localisation_files(self.cfg.src_dir)
                    else:
                        files = collect_localisation_files_by_lang(self.cfg.src_dir, [self.cfg.src_lang], include_replace=False)
                self.log.emit(f"No {self.cfg.src_lang} files found, scanning all languages...")
            else:
                self.log.emit(f"Found {len(files)} {self.cfg.src_lang} localisation files.")
                if not self.cfg.include_replace:
                    self.log.emit("Excluding files from 'replace' folders.")
        else:
            if use_parallel and len(os.listdir(self.cfg.src_dir)) > 50:
                self.log.emit("Using parallel file collection for large mod...")
                files = collect_localisation_files_parallel(self.cfg.src_dir)
            else:
                if self.cfg.include_replace:
                    files = collect_localisation_files(self.cfg.src_dir)
                else:
                    files = collect_localisation_files_by_lang(self.cfg.src_dir, [self.cfg.src_lang], include_replace=False)
            
        total = len(files)
        fc = max(1, self.cfg.files_concurrency)
        idx_counter = 0
        stop_flag = False
        
        if total == 0:
            self.log.emit("No localisation files found for the specified source language")
            self.aborted.emit("No files to translate")
            return

        def process_one(path: str) -> Tuple[str, str, bool, Optional[str]]:
            try:
                if self._cancel_requested():
                    return (path, path, False, "Cancelled by user")

                relname = os.path.relpath(path, self.cfg.src_dir)
                out_path = compute_output_path(path, self.cfg)
                if self.cfg.skip_existing and os.path.exists(out_path) and not self.cfg.in_place:
                    return (relname, out_path, True, None)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
                    if os.path.getsize(path) > 1024 * 1024:  # Files larger than 1MB
                        lines = []
                        while True:
                            chunk = f.readlines(10000)  # Read 10000 lines at a time
                            if not chunk:
                                break
                            lines.extend(chunk)
                    else:
                        lines = f.readlines()
                
                if self._cancel_requested():
                    return (path, out_path, False, "Cancelled by user")
                    
                prev_map: Dict[str, Tuple[str, str]] = {}
                if self.cfg.reuse_prev_loc and self.cfg.prev_loc_dir:
                    pf = _find_prev_localized_file(self.cfg.prev_loc_dir, relname, self.cfg.dst_lang)
                    if pf:
                        prev_map = _build_prev_map(pf)
                new_lines = self._process_file_lines(lines, backend, relname, prev_map)

                if self._cancel_requested():
                    return (path, out_path, False, "Cancelled by user")

                with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
                    f.writelines(new_lines)
                return (relname, out_path, False, None)
            except Exception as e:
                return (path, path, False, f"{e}\n{traceback.format_exc()}")

        try:
            with ThreadPoolExecutor(max_workers=fc, thread_name_prefix="file") as ex:
                pending_files = iter(files)
                in_flight = set()

                while len(in_flight) < fc:
                    try:
                        in_flight.add(ex.submit(process_one, next(pending_files)))
                    except StopIteration:
                        break

                while in_flight:
                    if self._cancel_requested() or not self._check_paused():
                        stop_flag = True
                        break

                    done, in_flight = wait(in_flight, timeout=0.1, return_when=FIRST_COMPLETED)
                    if not done:
                        continue

                    for fut in done:
                        rel, out_path, skipped, err = fut.result()
                        idx_counter += 1
                        if err is None:
                            if skipped:
                                self.log.emit(f"[SKIP] {out_path}")
                            else:
                                self.log.emit(f"[OK] → {out_path}")
                            self._files_done += 1
                        elif "cancel" in err.lower():
                            stop_flag = True
                        else:
                            self.log.emit(f"[ERR] {rel}: {err}")
                            self._files_done += 1

                        self.progress.emit(idx_counter, total)

                        if not stop_flag and not self._cancel_requested():
                            try:
                                in_flight.add(ex.submit(process_one, next(pending_files)))
                            except StopIteration:
                                pass

                    if stop_flag:
                        break
        except Exception as e:
            self.aborted.emit(f"Executor failed: {e}\n{traceback.format_exc()}")
            return

        self._cache.save()
        self._log_performance_summary()

        # Record cost for the session
        if self._translated_keys > 0:
            self._record_session_cost()

        if stop_flag:
            self.aborted.emit("Cancelled by user")
        else:
            self.finished_ok.emit()

    def _process_file_lines(self, lines: List[str], backend: TranslationBackend, relname: str,
                             prev_map: Dict[str, Tuple[str, str]]) -> List[str]:
        if self.cfg.batch_translation:
            return self._process_file_lines_batch(lines, backend, relname, prev_map)

        out: List[Optional[str]] = []
        header_replaced = False
        key_count = sum(1 for ln in lines if LOCALISATION_LINE_RE.match(ln))
        done_keys = 0
        self.file_progress.emit(relname)
        self.file_inner_progress.emit(0, max(1, key_count))

        is_google = isinstance(backend, GoogleFreeBackend)
        batch_size = max(1, self.cfg.batch_size) if getattr(backend, "supports_batch", False) else 1
        key_skip_re = re.compile(self.cfg.key_skip_regex) if self.cfg.key_skip_regex else None

        pending_items: List[Dict[str, object]] = []

        def flush_pending() -> None:
            nonlocal done_keys
            if not pending_items or self._cancel_requested():
                return

            texts = [item["masked"] for item in pending_items]
            translations = self._timed_translate_many(backend, texts)
            cache_updates: Dict[str, str] = {}

            for item, translated_text in zip(pending_items, translations):
                raw = _cleanup_llm_output(translated_text)
                if self.cfg.strip_md:
                    raw = _strip_model_noise(raw)
                candidate, cache_ok = self._finalize_candidate(item, raw)
                if not cache_ok:
                    self.log.emit(f"[WARN] Bad output for key '{item['key']}', keeping original.")

                out[item["out_index"]] = self._format_output_line(
                    item["pre"], item["key"], item["version"], candidate, item["post"]
                )
                if cache_ok:
                    cache_updates[item["masked"]] = candidate
                    self._translated_keys += 1

                done_keys += 1
                self._keys += 1
                if self._keys % (20 if is_google else 10) == 0:
                    self.stats.emit(self._words, self._keys, self._files_done)
                self.file_inner_progress.emit(done_keys, max(1, key_count))

            self._bulk_cache_set(cache_updates)
            pending_items.clear()

        for line in lines:
            if self._cancel_requested():
                break
            if not self._check_paused():
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
                self._metrics["cache_misses"] += 1
                out_index = len(out)
                out.append(None)
                pending_items.append({
                    "out_index": out_index,
                    "pre": pre,
                    "key": key,
                    "version": version,
                    "text": text,
                    "post": post,
                    "masked": masked,
                    "mapping": mapping,
                    "idx_tokens": idx_tokens,
                    "glmap": glmap,
                })
                if len(pending_items) >= batch_size:
                    flush_pending()
            else:
                self._metrics["cache_hits"] += 1
                candidate = _unmask_glossary(cached, glmap)
                candidate = _apply_replacements(candidate, self._glossary)
                if not _validate_translation(candidate, idx_tokens):
                    candidate = text
                out.append(self._format_output_line(pre, key, version, candidate, post))
                done_keys += 1
                self._keys += 1
                if self._keys % (20 if is_google else 10) == 0:
                    self.stats.emit(self._words, self._keys, self._files_done)
                self.file_inner_progress.emit(done_keys, max(1, key_count))

        flush_pending()
        self.stats.emit(self._words, self._keys, self._files_done)
        return [line if line is not None else "" for line in out]

    def _process_file_lines_batch(self, lines: List[str], backend: TranslationBackend, relname: str,
                                  prev_map: Dict[str, Tuple[str, str]]) -> List[str]:
        """Process file lines using structured batch mode with cached fallbacks."""
        processed_lines = lines[:]
        header_replaced = False
        key_count = sum(1 for ln in lines if LOCALISATION_LINE_RE.match(ln))
        done_keys = 0
        self.file_progress.emit(relname)
        self.file_inner_progress.emit(0, max(1, key_count))
        key_skip_re = re.compile(self.cfg.key_skip_regex) if self.cfg.key_skip_regex else None

        translatable_lines: List[Dict[str, object]] = []

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
                continue

            pre, key, version, text, post = m.groups()

            if key_skip_re and key_skip_re.search(key):
                continue

            if self.cfg.reuse_prev_loc and key in prev_map:
                prev_text, prev_post = prev_map[key]
                post2 = _combine_post_with_loc(prev_post or post or "", True)
                processed_lines[idx] = f"{pre}{key}:{version or 0} \"{prev_text}\"{post2}\n"
                done_keys += 1
                self._keys += 1
                self.file_inner_progress.emit(done_keys, max(1, key_count))
                continue

            masked, mapping, idx_tokens = mask_tokens(text)
            masked, glmap = _mask_glossary(masked, self._glossary)
            self._words += count_words_for_stats(masked)
            translatable_lines.append({
                "idx": idx,
                "pre": pre,
                "key": key,
                "version": version,
                "text": text,
                "post": post,
                "masked": masked,
                "mapping": mapping,
                "idx_tokens": idx_tokens,
                "glmap": glmap,
            })

        cache_map = self._bulk_cache_get([item["masked"] for item in translatable_lines])
        unresolved_items: List[Dict[str, object]] = []

        for item in translatable_lines:
            cached = cache_map.get(item["masked"])
            if cached is None:
                self._metrics["cache_misses"] += 1
                unresolved_items.append(item)
                continue

            self._metrics["cache_hits"] += 1
            candidate = _unmask_glossary(cached, item["glmap"])
            candidate = _apply_replacements(candidate, self._glossary)
            if not _validate_translation(candidate, item["idx_tokens"]):
                candidate = item["text"]
            processed_lines[item["idx"]] = self._format_output_line(
                item["pre"], item["key"], item["version"], candidate, item["post"]
            )
            done_keys += 1
            self._keys += 1
            self.file_inner_progress.emit(done_keys, max(1, key_count))

        chunk_size = max(1, self.cfg.chunk_size)

        def apply_chunk_results(chunk_items: List[Dict[str, object]], translated_texts: List[str]) -> None:
            nonlocal done_keys
            cache_updates: Dict[str, str] = {}
            for item, translated_text in zip(chunk_items, translated_texts):
                raw = _cleanup_llm_output(translated_text)
                if self.cfg.strip_md:
                    raw = _strip_model_noise(raw)
                candidate, cache_ok = self._finalize_candidate(item, raw)
                processed_lines[item["idx"]] = self._format_output_line(
                    item["pre"], item["key"], item["version"], candidate, item["post"]
                )
                if cache_ok:
                    cache_updates[item["masked"]] = candidate
                    self._translated_keys += 1
                done_keys += 1
                self._keys += 1
                self.file_inner_progress.emit(done_keys, max(1, key_count))
                if self._keys % 10 == 0:
                    self.stats.emit(self._words, self._keys, self._files_done)
            self._bulk_cache_set(cache_updates)

        for i in range(0, len(unresolved_items), chunk_size):
            if self._cancel_requested():
                self.log.emit("[CANCEL] Translation cancelled during batch processing")
                break
            if not self._check_paused():
                break
            chunk = unresolved_items[i:i + chunk_size]
            if not chunk:
                continue

            self.file_inner_progress.emit(done_keys, max(1, key_count))
            self.stats.emit(self._words, self._keys, self._files_done)
            chunk_texts = [item["masked"] for item in chunk]

            try:
                if getattr(backend, "supports_structured_batch", False):
                    batch_payload = batch_wrap_with_markers({item["key"]: item["masked"] for item in chunk})
                    expected_keys = [item["key"] for item in chunk]
                    response = self._timed_translate_structured_batch(backend, batch_payload, len(chunk))
                    translations = parse_batch_response(response, expected_keys)
                    if len(translations) != len(chunk):
                        self._metrics["structured_batch_failures"] += 1
                        self._metrics["structured_batch_fallbacks"] += 1
                        self.log.emit(
                            f"[WARN] Structured batch parse failed for {relname}: "
                            f"expected {len(chunk)} translations, got {len(translations)}. Falling back to translate_many."
                        )
                        translated_texts = self._timed_translate_many(backend, chunk_texts)
                    else:
                        translated_texts = [translations[item["key"]] for item in chunk]
                else:
                    translated_texts = self._timed_translate_many(backend, chunk_texts)

                apply_chunk_results(chunk, translated_texts)
            except Exception as e:
                self.log.emit(f"[ERR] Batch translation failed for {relname}: {e}")
                apply_chunk_results(chunk, self._timed_translate_many(backend, chunk_texts))

        self.stats.emit(self._words, self._keys, self._files_done)
        return processed_lines


class RetranslateWorker(QThread):
    progress = Signal(int, int)  # current, total
    translation_done = Signal(list)  # list of {'key': str, 'translation': str, 'row': int}
    log = Signal(str)

    def __init__(self, cfg: JobConfig, items: List[Dict]):
        super().__init__()
        self.cfg = cfg
        self.items = items
        self._cancel = False
        self._paused_event = threading.Event()
        self._paused_event.set()

    def cancel(self):
        self._cancel = True
        self._paused_event.set()

    def pause(self):
        self._paused_event.clear()

    def resume(self):
        self._paused_event.set()

    def is_paused(self) -> bool:
        return not self._paused_event.is_set()

    def _check_paused(self) -> bool:
        while not self._paused_event.wait(timeout=0.1):
            if self._cancel or self.isInterruptionRequested():
                return False
        return not self._cancel and not self.isInterruptionRequested()

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
            elif self.cfg.model_key == "Mistral AI":
                os.environ["MISTRAL_API_KEY"] = (self.cfg.mistral_api_key or "")
                os.environ["MISTRAL_MODEL"] = (self.cfg.mistral_model or "mistral-small-latest")
                os.environ["MISTRAL_TEMP"] = str(self.cfg.temperature)
                os.environ["MISTRAL_ASYNC"] = "1" if self.cfg.mistral_async else "0"
                os.environ["MISTRAL_CONCURRENCY"] = str(self.cfg.mistral_concurrency)
            elif self.cfg.model_key == "Nvidia NIM":
                os.environ["NVIDIA_API_KEY"] = (self.cfg.nvidia_api_key or "")
                os.environ["NVIDIA_MODEL"] = (self.cfg.nvidia_model or "moonshotai/kimi-k2.5")
                os.environ["NVIDIA_BASE_URL"] = (self.cfg.nvidia_base_url or "https://integrate.api.nvidia.com/v1/chat/completions")
                os.environ["NVIDIA_TEMP"] = str(self.cfg.temperature)
                os.environ["NVIDIA_ASYNC"] = "1" if self.cfg.nvidia_async else "0"
                os.environ["NVIDIA_CONCURRENCY"] = str(self.cfg.nvidia_concurrency)

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
                if self._cancel or self.isInterruptionRequested():
                    break
                if not self._check_paused():
                    break
                key = item['key']
                original = item['original']
                row = item['row']

                try:
                    masked, mapping, idx_tokens = mask_tokens(original)
                    masked, glmap = _mask_glossary(masked, glossary)

                    cached = cache.get(masked)
                    if cached is None:
                        if self._cancel or self.isInterruptionRequested():
                            break
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
    ok = Signal(str)
    fail = Signal(str)

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
                 mistral_api_key: Optional[str], mistral_model: Optional[str], mistral_async: bool, mistral_concurrency: int,
                 nvidia_api_key: Optional[str], nvidia_model: Optional[str], nvidia_base_url: str, nvidia_async: bool, nvidia_concurrency: int,
                 sqlite_cache_extension: str = ".db",
                 game_id: str = "hoi4",
                 mod_theme: Optional[str] = None):
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
        self.mistral_api_key = mistral_api_key
        self.mistral_model = mistral_model
        self.mistral_async = mistral_async
        self.mistral_concurrency = mistral_concurrency
        self.nvidia_api_key = nvidia_api_key
        self.nvidia_model = nvidia_model
        self.nvidia_base_url = nvidia_base_url
        self.nvidia_async = nvidia_async
        self.nvidia_concurrency = nvidia_concurrency
        self.game_id = game_id
        self.mod_theme = mod_theme

    def run(self):
        try:
            # Setup game context for translation
            os.environ["GAME_ID"] = (self.game_id or "hoi4")
            if self.mod_theme:
                os.environ["MOD_THEME"] = self.mod_theme
            else:
                os.environ["MOD_THEME"] = ""
            
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
            elif self.model_key == "Mistral AI":
                os.environ["MISTRAL_API_KEY"] = (self.mistral_api_key or "")
                os.environ["MISTRAL_MODEL"] = (self.mistral_model or "mistral-small-latest")
                os.environ["MISTRAL_TEMP"] = str(self.temperature)
                os.environ["MISTRAL_ASYNC"] = "1" if self.mistral_async else "0"
                os.environ["MISTRAL_CONCURRENCY"] = str(self.mistral_concurrency)
            elif self.model_key == "Nvidia NIM":
                os.environ["NVIDIA_API_KEY"] = (self.nvidia_api_key or "")
                os.environ["NVIDIA_MODEL"] = (self.nvidia_model or "moonshotai/kimi-k2.5")
                os.environ["NVIDIA_BASE_URL"] = (self.nvidia_base_url or "https://integrate.api.nvidia.com/v1/chat/completions")
                os.environ["NVIDIA_TEMP"] = str(self.temperature)
                os.environ["NVIDIA_ASYNC"] = "1" if self.nvidia_async else "0"
                os.environ["NVIDIA_CONCURRENCY"] = str(self.nvidia_concurrency)

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
                "Mistral AI": "mistral",
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
