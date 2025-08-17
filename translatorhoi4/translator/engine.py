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
    HF_AMD_OSS120B_Chat,
    HF_Yuntian_ChatGPT5Mini,
    HF_AMD_Llama4_17B,
    G4F_Backend,
    _cleanup_llm_output,
    _strip_model_noise,
)
from .cache import DiskCache
from .glossary import Glossary, _apply_replacements, _mask_glossary, _unmask_glossary
from .mask import mask_tokens, unmask_tokens, count_words_for_stats, _looks_like_http_error
from ..parsers.paradox_yaml import LOCALISATION_LINE_RE, HEADER_RE, SUPPORTED_LANG_HEADERS
from ..utils.fs import (
    collect_localisation_files,
    compute_output_path,
    _find_prev_localized_file,
    _build_prev_map,
    _combine_post_with_loc,
)


@dataclass
class JobConfig:
    src_dir: str
    out_dir: str
    src_lang: str
    dst_lang: str
    model_key: str
    temperature: float
    hf_token: Optional[str]
    hf_direct_url: Optional[str]
    in_place: bool
    skip_existing: bool
    strip_md: bool
    batch_size: int
    rename_files: bool
    hf_concurrency: int
    files_concurrency: int
    key_skip_regex: Optional[str]
    cache_path: Optional[str]
    glossary_path: Optional[str]
    prev_loc_dir: Optional[str]
    reuse_prev_loc: bool
    mark_loc_flag: bool
    g4f_model: Optional[str]
    g4f_provider: Optional[str]
    g4f_api_key: Optional[str]
    g4f_proxies: Optional[str]
    g4f_async: bool
    g4f_concurrency: int
    g4f_web_search: bool


MODEL_REGISTRY: Dict[str, callable] = {
    "Google (free unofficial)": lambda: GoogleFreeBackend(),
    "HF: amd/gpt-oss-120b-chatbot": lambda: HF_AMD_OSS120B_Chat(),
    "HF: yuntian-deng/ChatGPT (5 mini)": lambda: HF_Yuntian_ChatGPT5Mini(),
    "HF: amd/llama4-maverick-17b-128e-mi-amd": lambda: HF_AMD_Llama4_17B(),
    "G4F: chat.completions": lambda: G4F_Backend(
        model=os.environ.get("G4F_MODEL", "gemini-2.5-flash"),
        provider=os.environ.get("G4F_PROVIDER") or None,
        api_key=os.environ.get("G4F_API_KEY") or None,
        proxies=os.environ.get("G4F_PROXIES") or None,
        temperature=float(os.environ.get("G4F_TEMP", "0.7")),
        async_mode=(os.environ.get("G4F_ASYNC", "1") == "1"),
        concurrency=int(os.environ.get("G4F_CONCURRENCY", "6")),
        web_search=(os.environ.get("G4F_WEB_SEARCH", "0") == "1"),
        max_retries=int(os.environ.get("G4F_RETRIES", "4")),
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
        self._cache = DiskCache(cfg.cache_path or os.path.join(cfg.out_dir or cfg.src_dir, ".hoi4loc_cache.json"))
        if cfg.glossary_path and os.path.isfile(cfg.glossary_path):
            try:
                self._glossary = Glossary.load_csv(cfg.glossary_path)
            except Exception:
                pass

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self.cfg.model_key == "G4F: chat.completions":
                os.environ["G4F_MODEL"] = (self.cfg.g4f_model or "gemini-2.5-flash")
                os.environ["G4F_PROVIDER"] = (self.cfg.g4f_provider or "")
                os.environ["G4F_API_KEY"] = (self.cfg.g4f_api_key or "")
                os.environ["G4F_PROXIES"] = (self.cfg.g4f_proxies or "")
                os.environ["G4F_TEMP"] = str(self.cfg.temperature)
                os.environ["G4F_ASYNC"] = "1" if self.cfg.g4f_async else "0"
                os.environ["G4F_CONCURRENCY"] = str(self.cfg.g4f_concurrency)
                os.environ["G4F_WEB_SEARCH"] = "1" if self.cfg.g4f_web_search else "0"
            backend = MODEL_REGISTRY[self.cfg.model_key]()
            if hasattr(backend, 'temperature'):
                backend.temperature = self.cfg.temperature
            backend.set_token(self.cfg.hf_token)
            backend.set_direct_url(self.cfg.hf_direct_url)
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
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
                    lines = f.readlines()
                prev_map: Dict[str, Tuple[str, str]] = {}
                if self.cfg.reuse_prev_loc and self.cfg.prev_loc_dir:
                    pf = _find_prev_localized_file(self.cfg.prev_loc_dir, relname, self.cfg.dst_lang)
                    if pf:
                        prev_map = _build_prev_map(pf)
                new_lines = self._process_file_lines(lines, backend, relname, prev_map)
                with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
                    f.writelines(new_lines)
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
        if stop_flag:
            self.aborted.emit("Cancelled by user")
        else:
            self.finished_ok.emit()

    def _process_file_lines(self, lines: List[str], backend: TranslationBackend, relname: str,
                             prev_map: Dict[str, Tuple[str, str]]) -> List[str]:
        out: List[str] = []
        header_replaced = False
        key_count = sum(1 for ln in lines if LOCALISATION_LINE_RE.match(ln))
        done_keys = 0
        self.file_progress.emit(relname)
        self.file_inner_progress.emit(0, max(1, key_count))
        is_google = isinstance(backend, GoogleFreeBackend)
        is_g4f = isinstance(backend, G4F_Backend)
        batch_size = max(1, self.cfg.batch_size) if (is_google or is_g4f) else 1
        key_skip_re = re.compile(self.cfg.key_skip_regex) if self.cfg.key_skip_regex else None

        def _valid_after_unmask(candidate: str, idx_tokens: List[str]) -> bool:
            if _looks_like_http_error(candidate):
                return False
            if '<<SEG' in candidate or '<<END' in candidate or '__TKN' in candidate:
                return False
            for tok in idx_tokens:
                if tok and tok not in candidate:
                    return False
            return True

        if is_google or is_g4f:
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
                        if not _valid_after_unmask(candidate, idx_tokens):
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
                    if not _valid_after_unmask(candidate, idx_tokens):
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


class TestModelWorker(QThread):
    ok = pyqtSignal(str)
    fail = pyqtSignal(str)

    def __init__(self, model_key: str, src_lang: str, dst_lang: str, temperature: float,
                 hf_token: Optional[str], hf_direct_url: Optional[str], strip_md: bool, glossary_path: Optional[str],
                 g4f_model: Optional[str], g4f_provider: Optional[str], g4f_api_key: Optional[str], g4f_proxies: Optional[str],
                 g4f_async: bool, g4f_concurrency: int, g4f_web_search: bool):
        super().__init__()
        self.model_key = model_key
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.temperature = temperature
        self.hf_token = hf_token
        self.hf_direct_url = hf_direct_url
        self.strip_md = strip_md
        self.glossary_path = glossary_path
        self.g4f_model = g4f_model
        self.g4f_provider = g4f_provider
        self.g4f_api_key = g4f_api_key
        self.g4f_proxies = g4f_proxies
        self.g4f_async = g4f_async
        self.g4f_concurrency = g4f_concurrency
        self.g4f_web_search = g4f_web_search

    def run(self):
        try:
            gl = Glossary([], {})
            if self.glossary_path and os.path.isfile(self.glossary_path):
                gl = Glossary.load_csv(self.glossary_path)
            if self.model_key == "G4F: chat.completions":
                os.environ["G4F_MODEL"] = (self.g4f_model or "gemini-2.5-flash")
                os.environ["G4F_PROVIDER"] = (self.g4f_provider or "")
                os.environ["G4F_API_KEY"] = (self.g4f_api_key or "")
                os.environ["G4F_PROXIES"] = (self.g4f_proxies or "")
                os.environ["G4F_TEMP"] = str(self.temperature)
                os.environ["G4F_ASYNC"] = "1" if self.g4f_async else "0"
                os.environ["G4F_CONCURRENCY"] = str(self.g4f_concurrency)
                os.environ["G4F_WEB_SEARCH"] = "1" if self.g4f_web_search else "0"
            backend = MODEL_REGISTRY[self.model_key]()
            if hasattr(backend, 'temperature'):
                backend.temperature = self.temperature
            backend.set_token(self.hf_token)
            backend.set_direct_url(self.hf_direct_url)
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
