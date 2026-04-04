"""Yandex Translate and Yandex Cloud translation backends."""
from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from typing import List, Optional

from ..mask import _extract_marked, _looks_like_http_error
from ..prompts import system_prompt, wrap_with_markers
from .base import TranslationBackend, _extract_text, _cleanup_llm_output, _strip_model_noise


class YandexTranslateBackend(TranslationBackend):
    name = "Yandex Translate"
    supports_batch = True
    supports_system_prompt = False

    def __init__(self, api_key: Optional[str], iam_token: Optional[str] = None, folder_id: str = "",
                 temperature: float = 0.0, async_mode: bool = False, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.iam_token = (iam_token or "").strip() or None
        self.folder_id = folder_id
        self.temperature = 0.0
        self.async_mode = False
        self.concurrency = 1
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

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_session()
        target_language = dst_lang.lower()[:2]  # e.g., 'ru', 'en'
        texts = [text]
        body = {
            "targetLanguageCode": target_language,
            "texts": texts,
            "folderId": self.folder_id,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        elif self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    "https://translate.api.cloud.yandex.net/translate/v2/translate",
                    json=body,
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    translations = data.get("translations", [])
                    if translations:
                        return translations[0].get("text", text)
            except Exception:
                pass
            time.sleep(delay)
            delay = min(5.0, delay * 1.7)
        return text

    def translate_many(self, texts: List[str], src_lang: str, dst_lang: str) -> List[str]:
        # Yandex supports batch in one request
        self._init_session()
        target_language = dst_lang.lower()[:2]
        body = {
            "targetLanguageCode": target_language,
            "texts": texts,
            "folderId": self.folder_id,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        elif self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    "https://translate.api.cloud.yandex.net/translate/v2/translate",
                    json=body,
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    translations = data.get("translations", [])
                    return [t.get("text", orig) for t, orig in zip(translations, texts)]
            except Exception:
                pass
            time.sleep(delay)
            delay = min(5.0, delay * 1.7)
        return texts


class YandexCloudBackend(TranslationBackend):
    name = "Yandex Cloud"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str = "aliceai-llm/latest", folder_id: str = "",
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = model
        self.folder_id = folder_id
        self.base_url = "https://rest-assistant.api.cloud.yandex.net/v1"
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
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, project=self.folder_id)

    def _init_async(self):
        with self._init_lock:
            if self._aclient is not None and self._loop is not None:
                return
            import openai
            self._aclient = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, project=self.folder_id)
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
        game_id = os.environ.get("GAME_ID", "hoi4")
        mod_theme = os.environ.get("MOD_THEME", "")
        sys_prompt = system_prompt(src_lang, dst_lang, game_id, mod_theme if mod_theme else None)
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                res = self._client.responses.create(  # Note: Yandex uses responses.create, not chat.completions
                    model=f"gpt://{self.folder_id}/{self.model}",
                    temperature=self.temperature,
                    instructions=sys_prompt,
                    input=payload,
                    max_output_tokens=500
                )
                content = res.output_text if hasattr(res, "output_text") else _extract_text(res)
                if not _looks_like_http_error(content):
                    raw = _strip_model_noise(_cleanup_llm_output(content))
                    return _extract_marked(raw, sid)
            except Exception:
                pass
            time.sleep(delay)
            delay = min(5.0, delay * 1.7)
        return text

    async def _async_translate_one(self, aclient, sem, text: str, src_lang: str, dst_lang: str):
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        game_id = os.environ.get("GAME_ID", "hoi4")
        mod_theme = os.environ.get("MOD_THEME", "")
        sys_prompt = system_prompt(src_lang, dst_lang, game_id, mod_theme if mod_theme else None)
        delay = 0.3
        for attempt in range(self.max_retries):
            async with sem:
                try:
                    res = await aclient.responses.create(
                        model=f"gpt://{self.folder_id}/{self.model}",
                        temperature=self.temperature,
                        instructions=sys_prompt,
                        input=payload,
                        max_output_tokens=500
                    )
                    content = res.output_text if hasattr(res, "output_text") else _extract_text(res)
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
