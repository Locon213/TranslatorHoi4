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

    def __init__(self):
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
                 max_retries: int = 4):
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
                 concurrency: int = 6, max_retries: int = 4):
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
                 max_retries: int = 4):
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
                 max_retries: int = 4):
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


# --- OpenAI Compatible Backend ---

class OpenAICompatibleBackend(TranslationBackend):
    name = "OpenAI Compatible API"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, api_key: Optional[str], model: str, base_url: str,
                 temperature: float = 0.7, async_mode: bool = True,
                 concurrency: int = 6, max_retries: int = 4):
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