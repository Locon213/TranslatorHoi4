"""Translation backends (ported from monolith)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from typing import List, Optional

from .mask import _extract_marked, _looks_like_http_error
from .prompts import system_prompt, wrap_with_markers


# --- helpers ---

def _extract_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (int, float, bool)):
        return str(result)
    if isinstance(result, (list, tuple)):
        return _extract_text(result[0]) if result else ""
    if isinstance(result, dict):
        for k in ("text", "response", "Response", "data", "message", "content"):
            if k in result:
                return _extract_text(result[k])
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)
    return str(result)


def _cleanup_llm_output(text: str) -> str:
    t = text.strip()
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1]
    if t.startswith("'''") and t.endswith("'''"):
        t = t[3:-3]
    if t.startswith("```") and t.endswith("```"):
        t = t.strip('`')
        lines = t.splitlines()
        if lines and not lines[0].strip().startswith('{'):
            t = "\n".join(lines[1:])
    return t.strip()


def _strip_model_noise(text: str) -> str:
    t = text.strip()
    # Remove reasoning tags like <think>...</think>
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
    if t.startswith("```") and t.endswith("```"):
        t = t.strip('`')
        lines = t.splitlines()
        if lines and not lines[0].strip().startswith('{'):
            t = "\n".join(lines[1:])
    idx = t.lower().rfind("response:")
    if idx != -1:
        t = t[idx + len("response:"):].strip()
    t = re.sub(r'^\s*\*\*.*?\*\*:\s*$', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*-{3,}\s*$', '', t, flags=re.MULTILINE)
    lines = [ln for ln in t.splitlines() if not re.match(r'^\s*[\-\*\d+\.]\s', ln)]
    if lines:
        t = "\n".join(lines).strip()
    return t.strip().strip('*').strip()


# --- Backends ---

class TranslationBackend:
    name = "Base"
    supports_system_prompt = False
    supports_batch = False

    def __init__(self, rpm_limit: int = 60):
        self.rpm_limit = rpm_limit
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()

    def _rate_limit(self):
        if self.rpm_limit <= 0:
            return
        min_interval = 60.0 / self.rpm_limit
        with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        raise NotImplementedError

    def translate_many(self, texts: List[str], src_lang: str, dst_lang: str) -> List[str]:
        return [self.translate(t, src_lang, dst_lang) for t in texts]

    def warmup(self) -> None:
        pass

    def set_token(self, token: Optional[str]):
        pass

    def set_direct_url(self, url: Optional[str]):
        pass


class GoogleFreeBackend(TranslationBackend):
    name = "Google (free unofficial)"
    supports_batch = True

    def __init__(self, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self._mode = None
        self._gt = None
        self._dt = None
        try:
            from deep_translator import GoogleTranslator as DTGoogle
            self._dt = DTGoogle; self._mode = "deep"
        except Exception:
            self._dt = None
        try:
            from googletrans import Translator
            self._gt = Translator
            if self._mode is None:
                self._mode = "googletrans"
        except Exception:
            self._gt = None
        if self._mode is None:
            raise RuntimeError("Установи deep-translator или googletrans.")
        self._client = self._dt(source='auto', target='ru') if self._mode == "deep" else self._gt()

    def _dst(self, dst_lang: str) -> str:
        return 'ru' if dst_lang.startswith('ru') else dst_lang[:2]

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        dst = self._dst(dst_lang)
        if self._mode == "deep":
            try:
                return self._client.translate(text, target=dst)
            except Exception:
                self._client = self._dt(source='auto', target=dst)
                return self._client.translate(text, target=dst)
        else:
            try:
                res = self._client.translate(text, src=src_lang[:2], dest=dst)
                import asyncio
                if asyncio.iscoroutine(res):
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        res_obj = loop.run_until_complete(res)
                    finally:
                        try:
                            loop.stop()
                        except Exception:
                            pass
                        loop.close()
                    return getattr(res_obj, "text", str(res_obj))
                return getattr(res, "text", str(res))
            except Exception as e:
                return str(e)

    def translate_many(self, texts: List[str], src_lang: str, dst_lang: str) -> List[str]:
        dst = self._dst(dst_lang)
        if self._mode == "deep":
            try:
                res = self._client.translate_batch(texts, target=dst)
                return list(res)
            except Exception:
                return [self.translate(t, src_lang, dst_lang) for t in texts]
        else:
            try:
                res_list = self._client.translate(texts, src=src_lang[:2], dest=dst)
                import asyncio
                if hasattr(res_list, '__await__'):
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        out = loop.run_until_complete(res_list)
                    finally:
                        try:
                            loop.stop()
                        except Exception:
                            pass
                        loop.close()
                    return [getattr(r, "text", str(r)) for r in out]
                return [getattr(r, "text", str(r)) for r in res_list]
            except Exception:
                return [self.translate(t, src_lang, dst_lang) for t in texts]


# --- G4F Backend (Updated to Official API) ---

class G4F_Backend(TranslationBackend):
    name = "G4F: API (g4f.dev)"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str,
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = (model or "gpt-4o").strip()
        self.base_url = "https://g4f.dev/v1"
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
            # Standard OpenAI client pointing to G4F URL
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


# --- IO Intelligence Backend ---

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


# --- Anthropic (Claude) Backend ---

class AnthropicBackend(TranslationBackend):
    name = "Anthropic: Claude"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str = "claude-sonnet-4-5-20250929",
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = model
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
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except Exception as e:
                raise RuntimeError(f"Anthropic не установлен: {e}")

    def _init_async(self):
        with self._init_lock:
            if self._aclient is not None and self._loop is not None:
                return
            try:
                from anthropic import AsyncAnthropic
                self._aclient = AsyncAnthropic(api_key=self.api_key)
            except Exception as e:
                raise RuntimeError(f"Anthropic не установлен: {e}")
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
        return [{"role": "user", "content": f"{sys_prompt}\n\n{payload}"}]

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_sync()
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        sys_prompt = system_prompt(src_lang, dst_lang)
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                res = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=self._messages(sys_prompt, payload),
                    temperature=self.temperature,
                )
                content = _extract_text(res)
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
        sys_prompt = system_prompt(src_lang, dst_lang)
        delay = 0.3
        for attempt in range(self.max_retries):
            async with sem:
                try:
                    res = await aclient.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=self._messages(sys_prompt, payload),
                        temperature=self.temperature,
                    )
                    content = _extract_text(res)
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


# --- Gemini Backend ---

class GeminiBackend(TranslationBackend):
    name = "Google: Gemini"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str = "gemini-2.5-flash",
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = model
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
            try:
                from google import genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except Exception as e:
                raise RuntimeError(f"Google GenAI не установлен: {e}")

    def _init_async(self):
        with self._init_lock:
            if self._aclient is not None and self._loop is not None:
                return
            try:
                from google import genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                self._aclient = genai.GenerativeModel(self.model)
            except Exception as e:
                raise RuntimeError(f"Google GenAI не установлен: {e}")
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
        return f"{sys_prompt}\n\n{payload}"

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_sync()
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        sys_prompt = system_prompt(src_lang, dst_lang)
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                res = self._client.generate_content(
                    contents=self._messages(sys_prompt, payload),
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": 1024,
                    },
                )
                content = _extract_text(res)
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
        sys_prompt = system_prompt(src_lang, dst_lang)
        delay = 0.3
        for attempt in range(self.max_retries):
            async with sem:
                try:
                    res = await aclient.generate_content(
                        contents=self._messages(sys_prompt, payload),
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": 1024,
                        },
                    )
                    content = _extract_text(res)
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


# --- Yandex Translate Backend ---

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


# --- Yandex Cloud Backend ---

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
        sys_prompt = system_prompt(src_lang, dst_lang)
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
        sys_prompt = system_prompt(src_lang, dst_lang)
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


# --- DeepL Backend ---

class DeepLBackend(TranslationBackend):
    name = "DeepL API"
    supports_batch = True
    supports_system_prompt = False  # DeepL is direct translation, no system prompts

    def __init__(self, api_key: Optional[str], model: str = "",  # Model not used for DeepL
                 temperature: float = 0.0, async_mode: bool = False, concurrency: int = 6,  # Sync only for simplicity
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.temperature = 0.0  # Not applicable
        self.async_mode = False  # DeepL client is sync
        self.concurrency = 1  # Not async
        self.max_retries = max(1, int(max_retries))
        self._client = None
        self._init_lock = threading.Lock()

    def _init_client(self):
        with self._init_lock:
            if self._client is not None:
                return
            import deepl
            self._client = deepl.DeepLClient(self.api_key)

    def warmup(self):
        self._init_client()

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_client()
        # DeepL uses language codes like 'EN', 'RU', etc.
        src = src_lang.upper()[:2] if src_lang else None
        dst = dst_lang.upper()[:2]
        if dst == 'RU':
            dst = 'RU'
        elif dst == 'EN':
            dst = 'EN-US'  # or EN-GB, but default to US
        # Add more mappings if needed
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                result = self._client.translate_text(text, target_lang=dst, source_lang=src)
                return result.text
            except Exception:
                pass
            time.sleep(delay)
            delay = min(5.0, delay * 1.7)
        return text

    def translate_many(self, texts: List[str], src_lang: str, dst_lang: str) -> List[str]:
        # DeepL supports batch, but for simplicity, translate one by one
        return [self.translate(t, src_lang, dst_lang) for t in texts]


# --- Fireworks.ai Backend ---

class FireworksBackend(TranslationBackend):
    name = "Fireworks.ai"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = model
        self.base_url = "https://api.fireworks.ai/inference/v1"
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
        self._rate_limit()
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


# --- Groq Backend ---

class GroqBackend(TranslationBackend):
    name = "Groq"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str = "openai/gpt-oss-20b",
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = model
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
            from groq import Groq
            self._client = Groq(api_key=self.api_key)

    def _init_async(self):
        with self._init_lock:
            if self._aclient is not None and self._loop is not None:
                return
            from groq import AsyncGroq
            self._aclient = AsyncGroq(api_key=self.api_key)
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


# --- Together.ai Backend ---

class TogetherBackend(TranslationBackend):
    name = "Together.ai"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6,
                 max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = model
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
            from together import Together
            self._client = Together(api_key=self.api_key)

    def _init_async(self):
        with self._init_lock:
            if self._aclient is not None and self._loop is not None:
                return
            from together import AsyncTogether
            self._aclient = AsyncTogether(api_key=self.api_key)
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


# --- Ollama Backend ---

class OllamaBackend(TranslationBackend):
    name = "Ollama"
    supports_batch = True
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

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        self._rate_limit()
        self._init_session()
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        sys_prompt = system_prompt(src_lang, dst_lang)
        messages = self._messages(sys_prompt, payload)
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": self.temperature}
                    },
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

    async def _async_translate_one(self, session, sem, text: str, src_lang: str, dst_lang: str):
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        sys_prompt = system_prompt(src_lang, dst_lang)
        messages = self._messages(sys_prompt, payload)
        delay = 0.3
        for attempt in range(self.max_retries):
            async with sem:
                try:
                    import aiohttp
                    async with session.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "stream": False,
                            "options": {"temperature": self.temperature}
                        },
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


# --- OpenAI Compatible Backend ---

class OpenAICompatibleBackend(TranslationBackend):
    name = "OpenAI Compatible API"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str, base_url: str,
                 temperature: float = 0.7, async_mode: bool = True,
                 concurrency: int = 6, max_retries: int = 4, rpm_limit: int = 60):
        super().__init__(rpm_limit)
        self.api_key = (api_key or "").strip() or None
        self.model = (model or "gpt-4").strip()
        self.base_url = (base_url or "https://api.openai.com/v1/").strip()
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