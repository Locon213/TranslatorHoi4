"""IO Intelligence translation backend."""
from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from typing import List, Optional

from ..mask import _extract_marked, _looks_like_http_error
from ..prompts import system_prompt, wrap_with_markers
from .base import TranslationBackend, _extract_text, _cleanup_llm_output, _strip_model_noise


class IO_Intelligence_Backend(TranslationBackend):
    name = "IO: chat.completions"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str, base_url: str,
                 temperature: float = 0.7, async_mode: bool = True,
                 concurrency: int = 6, max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = (model or "meta-llama/Llama-3.3-70B-Instruct").strip()
        self.base_url = (base_url or "https://api.intelligence.io.solutions/api/v1/").strip()
        self.temperature = float(temperature)
        self.async_mode = bool(async_mode)
        self.concurrency = max(1, int(concurrency))
        self.max_retries = max(1, int(max_retries))
        self._client = None
        self._aclient = None
        self._init_lock = threading.Lock()
        self._loop = None

    def _init_sync(self):
        with self._init_lock:
            if self._client is not None:
                return
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _init_async(self):
        with self._init_lock:
            if self._aclient is not None and self._loop is not None:
                return
            import openai
            self._aclient = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            try:
                self._loop = asyncio.new_event_loop()
            except Exception:
                self._loop = asyncio.get_event_loop()

    def warmup(self):
        if self.async_mode:
            self._init_async()
        else:
            self._init_sync()

    def _messages(self, sys_prompt: str, payload: str):
        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": payload}]

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_sync()
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        sys_prompt = system_prompt(src_lang, dst_lang)
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                res = self._client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(sys_prompt, payload),
                    temperature=self.temperature,
                )
                content = res.choices[0].message.content if hasattr(res, "choices") else _extract_text(res)
                if not _looks_like_http_error(content):
                    raw = _strip_model_noise(_cleanup_llm_output(_extract_text(content)))
                    return _extract_marked(raw, sid)
            except Exception:
                pass
            time.sleep(delay)
            delay = min(5.0, delay * 1.7)
        return text

    async def _async_translate_one(self, aclient, sem, text: str, src_lang: str, dst_lang: str):
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        sys_prompt = system_prompt(src_lang, dst_lang)
        delay = 0.3
        for attempt in range(self.max_retries):
            async with sem:
                try:
                    res = await aclient.chat.completions.create(
                        model=self.model,
                        messages=self._messages(sys_prompt, payload),
                        temperature=self.temperature,
                    )
                    content = res.choices[0].message.content if hasattr(res, "choices") else _extract_text(res)
                    if not _looks_like_http_error(content):
                        raw = _strip_model_noise(_cleanup_llm_output(_extract_text(content)))
                        return _extract_marked(raw, sid)
                except Exception:
                    pass
            await asyncio.sleep(delay)
            delay = min(3.0, delay * 1.7)
        return text

    def translate_many(self, texts: List[str], src_lang: str, dst_lang: str) -> List[str]:
        if not self.async_mode:
            return [self.translate(t, src_lang, dst_lang) for t in texts]
        self._init_async()
        sem = asyncio.Semaphore(self.concurrency)

        async def runner():
            tasks = [self._async_translate_one(self._aclient, sem, t, src_lang, dst_lang) for t in texts]
            return await asyncio.gather(*tasks, return_exceptions=False)

        try:
            return self._loop.run_until_complete(runner())
        except Exception:
            try:
                return asyncio.run(runner())
            except Exception:
                return [self.translate(t, src_lang, dst_lang) for t in texts]
