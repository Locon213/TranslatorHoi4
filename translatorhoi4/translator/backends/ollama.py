"""Ollama translation backend."""
from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from typing import List, Optional

from ..mask import _extract_marked, _looks_like_http_error
from ..prompts import batch_system_prompt, system_prompt, wrap_with_markers
from .base import TranslationBackend, _extract_text, _cleanup_llm_output, _strip_model_noise


class OllamaBackend(TranslationBackend):
    name = "Ollama"
    supports_batch = True
    supports_structured_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None  # Not used, but for consistency
        self.model = (model or "llama3.2").strip()
        self.base_url = base_url.rstrip('/')
        self.temperature = float(temperature)
        self.async_mode = bool(async_mode)
        self.concurrency = max(1, int(concurrency))
        self.max_retries = max(1, int(max_retries))
        self._session = None
        self._init_lock = threading.Lock()

    def _init_session(self):
        with self._init_lock:
            if self._session is not None:
                return
            import requests
            self._session = requests.Session()

    def warmup(self):
        self._init_session()

    def _messages(self, sys_prompt: str, payload: str):
        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": payload}]

    def _single_request(self, text: str, src_lang: str, dst_lang: str):
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        game_id = os.environ.get("GAME_ID", "hoi4")
        mod_theme = os.environ.get("MOD_THEME", "")
        sys_prompt = system_prompt(src_lang, dst_lang, game_id, mod_theme if mod_theme else None)
        return sys_prompt, payload, sid

    def _batch_request(self, batch_payload: str, src_lang: str, dst_lang: str):
        game_id = os.environ.get("GAME_ID", "hoi4")
        mod_theme = os.environ.get("MOD_THEME", "")
        sys_prompt = batch_system_prompt(src_lang, dst_lang, game_id, mod_theme if mod_theme else None)
        return sys_prompt, batch_payload

    def _request_json(self, sys_prompt: str, payload: str):
        return {
            "model": self.model,
            "messages": self._messages(sys_prompt, payload),
            "stream": False,
            "options": {"temperature": self.temperature}
        }

    def translate_one(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_session()
        sys_prompt, payload, sid = self._single_request(text, src_lang, dst_lang)
        delay = 0.5
        for _ in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/api/chat",
                    json=self._request_json(sys_prompt, payload),
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    content = _extract_text(data)
                    if not _looks_like_http_error(content):
                        raw = _strip_model_noise(_cleanup_llm_output(content))
                        return _extract_marked(raw, sid)
            except Exception:
                pass
            time.sleep(delay)
            delay = min(5.0, delay * 1.7)
        return text

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        return self.translate_one(text, src_lang, dst_lang)

    def translate_structured_batch(self, batch_payload: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_session()
        sys_prompt, payload = self._batch_request(batch_payload, src_lang, dst_lang)
        delay = 0.5
        for _ in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/api/chat",
                    json=self._request_json(sys_prompt, payload),
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    content = _extract_text(data)
                    if not _looks_like_http_error(content):
                        return _strip_model_noise(_cleanup_llm_output(content))
            except Exception:
                pass
            time.sleep(delay)
            delay = min(5.0, delay * 1.7)
        return ""

    async def _async_translate_one(self, session, sem, text: str, src_lang: str, dst_lang: str):
        sys_prompt, payload, sid = self._single_request(text, src_lang, dst_lang)
        delay = 0.3
        for _ in range(self.max_retries):
            async with sem:
                try:
                    import aiohttp
                    async with session.post(
                        f"{self.base_url}/api/chat",
                        json=self._request_json(sys_prompt, payload),
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            content = _extract_text(data)
                            if not _looks_like_http_error(content):
                                raw = _strip_model_noise(_cleanup_llm_output(content))
                                return _extract_marked(raw, sid)
                except Exception:
                    pass
            await asyncio.sleep(delay)
            delay = min(3.0, delay * 1.7)
        return text

    def translate_many(self, texts: List[str], src_lang: str, dst_lang: str) -> List[str]:
        if not self.async_mode:
            return [self.translate(t, src_lang, dst_lang) for t in texts]
        import aiohttp
        sem = asyncio.Semaphore(self.concurrency)

        async def runner():
            async with aiohttp.ClientSession() as session:
                tasks = [self._async_translate_one(session, sem, t, src_lang, dst_lang) for t in texts]
                return await asyncio.gather(*tasks, return_exceptions=False)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(runner())
        except Exception:
            try:
                return asyncio.run(runner())
            except Exception:
                return [self.translate(t, src_lang, dst_lang) for t in texts]
