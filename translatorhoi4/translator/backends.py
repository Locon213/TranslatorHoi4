"""Translation backends (ported from monolith)."""
from __future__ import annotations

import asyncio, re
import hashlib
import json
import os
import tempfile
import threading
import time
from typing import Callable, Dict, List, Optional

from .mask import _extract_marked, _looks_like_http_error
from .prompts import system_prompt, wrap_with_markers


# --- HF no-auth env utilities ---

def _hf_force_no_auth_env():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        os.environ.pop(k, None)
    tmp_home = os.path.join(tempfile.gettempdir(), "hoi4_hf_home")
    tmp_cache = os.path.join(tempfile.gettempdir(), "hoi4_hf_cache")
    os.makedirs(tmp_home, exist_ok=True)
    os.makedirs(tmp_cache, exist_ok=True)
    os.environ.update({
        "HF_HOME": tmp_home,
        "HUGGINGFACE_HUB_CACHE": tmp_cache,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
    })


def _disable_gradio_auth_headers():
    try:
        import gradio_client.client as gcc
    except Exception:
        return
    if getattr(gcc.Client, "_hoi4_no_auth_patched", False):
        return
    def _get_headers_no_auth(self):
        return {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "User-Agent": "hoi4-localizer",
        }
    gcc.Client._get_headers = _get_headers_no_auth
    gcc.Client._hoi4_no_auth_patched = True


# --- helpers ---

def _is_valid_hf_token(tok: Optional[str]) -> bool:
    if not tok:
        return False
    if any(ch == '\x00' for ch in tok):
        return False
    s = tok.strip()
    if len(s) < 16:
        return False
    if not all(32 <= ord(ch) <= 126 for ch in s):
        return False
    return True


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


# --- HF Spaces ---

class HF_Base(TranslationBackend):
    def __init__(self):
        from gradio_client import Client
        _disable_gradio_auth_headers()
        self.Client = Client
        self.token: Optional[str] = None
        self.space: Optional[str] = None
        self.direct_url: Optional[str] = None
        self._tls = threading.local()

    def set_token(self, token: Optional[str]):
        self.token = token

    def set_direct_url(self, url: Optional[str]):
        self.direct_url = (url or "").strip() or None

    def _make_client(self, default_space: str):
        _hf_force_no_auth_env()
        tok = self.token if _is_valid_hf_token(self.token) else None
        if self.direct_url:
            return self.Client(self.direct_url, hf_token=None, headers={})
        if tok:
            try:
                return self.Client(default_space, hf_token=tok)
            except Exception:
                pass
        owner, space = default_space.split("/")
        direct = f"https://{owner.replace('_','-')}-{space.replace('_','-')}.hf.space"
        return self.Client(direct, hf_token=None, headers={})

    def _on_new_client(self, client):
        pass

    def _get_client(self):
        c = getattr(self._tls, "client", None)
        if c is None:
            c = self._make_client(self.space or "")
            self._on_new_client(c)
            self._tls.client = c
        return c

    def warmup(self):
        if self.space:
            _ = self._make_client(self.space)


class HF_AMD_OSS120B_Chat(HF_Base):
    name = "HF: amd/gpt-oss-120b-chatbot"
    supports_system_prompt = True

    def __init__(self, temperature: float = 0.7, token: Optional[str] = None):
        super().__init__()
        self.temperature = temperature
        self.token = token
        self.space = "amd/gpt-oss-120b-chatbot"

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        client = self._get_client()
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        sys_prompt = (
            "You are a specialised game localisation engine. "
            f"Translate from {src_lang} to {dst_lang}. "
            "Keep tokens intact: $VARS$, [Scripted.Macros], and \\n. "
            "Return ONLY the translated text between the SAME markers. "
            "DO NOT alter '<<SEG id>>' and '<<END id>>'."
        )
        res = client.predict(
            message=payload,
            system_prompt=sys_prompt,
            temperature=self.temperature,
            api_name="/chat",
        )
        raw = _strip_model_noise(_extract_text(res))
        return _extract_marked(raw, sid)


class HF_Yuntian_ChatGPT5Mini(HF_Base):
    name = "HF: yuntian-deng/ChatGPT (5 mini)"

    def __init__(self, temperature: float = 0.7, token: Optional[str] = None):
        super().__init__()
        self.temperature = temperature
        self.token = token
        self.space = "yuntian-deng/ChatGPT"

    def _on_new_client(self, client):
        try:
            _ = client.predict(api_name="/enable_inputs")
        except Exception:
            pass

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        client = self._get_client()
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = wrap_with_markers(text, sid)
        prompt = (
            f"Translate from {src_lang} to {dst_lang}. "
            "Keep tokens intact: $VARS$, [Root.GetX], \\n. "
            "Output ONLY the translation between the SAME markers. "
            "Do NOT change markers '<<SEG id>>' and '<<END id>>'.\n\n" + payload
        )
        res = client.predict(
            inputs=prompt,
            top_p=1,
            temperature=max(0.1, min(1.2, self.temperature)),
            chat_counter=0,
            chatbot=[],
            api_name="/predict",
        )
        raw = _strip_model_noise(_extract_text(res))
        return _extract_marked(raw, sid)


class HF_AMD_Llama4_17B(HF_Base):
    name = "HF: amd/llama4-maverick-17b-128e-mi-amd"
    supports_system_prompt = True

    def __init__(self, temperature: float = 0.3, token: Optional[str] = None, max_new_tokens: int = 512):
        super().__init__()
        self.temperature = temperature
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.space = "amd/llama4-maverick-17b-128e-mi-amd"

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        client = self._get_client()
        sid = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        payload = {"text": wrap_with_markers(text, sid), "files": []}
        sys_prompt = (
            "You are a professional game localizer for Paradox titles. "
            f"Translate from {src_lang} to {dst_lang}. Preserve $VARS$, [Scripted.Macros], and \\n. "
            "Return ONLY the translation between the SAME markers. "
            "Do NOT alter '<<SEG id>>' and '<<END id>>'."
        )
        res = client.predict(
            message=payload,
            param_2=sys_prompt,
            param_3=self.max_new_tokens,
            param_4=self.temperature,
            param_5=0,
            param_6=0,
            api_name="/chat",
        )
        raw = _strip_model_noise(_extract_text(res))
        return _extract_marked(raw, sid)


# --- G4F Backend ---

class G4F_Backend(TranslationBackend):
    name = "G4F: chat.completions"
    supports_batch = True
    supports_system_prompt = True

    def __init__(self, model: str, provider: Optional[str], api_key: Optional[str], proxies: Optional[str],
                 temperature: float = 0.7, async_mode: bool = True, concurrency: int = 6, web_search: bool = False,
                 max_retries: int = 4):
        self.model = (model or "gemini-2.5-flash").strip()
        self.provider_str = (provider or "").strip()
        self.api_key = (api_key or "").strip() or None
        self.proxies = (proxies or "").strip() or None
        self.temperature = float(temperature)
        self.async_mode = bool(async_mode)
        self.concurrency = max(1, int(concurrency))
        self.web_search = bool(web_search)
        self.max_retries = max(1, int(max_retries))
        self._client = None
        self._aclient = None
        self._init_lock = threading.Lock()
        self._loop = None

    def _get_sync_client_cls(self):
        from g4f.client import Client
        return Client

    def _get_async_client_cls(self):
        from g4f.client import AsyncClient
        return AsyncClient

    def _resolve_provider(self):
        if not self.provider_str:
            return None
        try:
            parts = self.provider_str.split('.')
            mod = __import__(parts[0])
            cur = mod
            for p in parts[1:]:
                cur = getattr(cur, p)
            return cur
        except Exception:
            return None

    def _init_sync(self):
        with self._init_lock:
            if self._client is not None:
                return
            try:
                Client = self._get_sync_client_cls()
            except Exception as e:
                raise RuntimeError(f"g4f не установлен: {e}")
            prov = self._resolve_provider()
            kwargs = {}
            if prov is not None:
                kwargs["provider"] = prov
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.proxies:
                kwargs["proxies"] = self.proxies
            self._client = Client(**kwargs)

    def _init_async(self):
        with self._init_lock:
            if self._aclient is not None and self._loop is not None:
                return
            try:
                AsyncClient = self._get_async_client_cls()
            except Exception as e:
                raise RuntimeError(f"g4f не установлен: {e}")
            prov = self._resolve_provider()
            kwargs = {}
            if prov is not None:
                kwargs["provider"] = prov
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.proxies:
                kwargs["proxies"] = self.proxies
            self._aclient = AsyncClient(**kwargs)
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
                    web_search=self.web_search,
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
                        web_search=self.web_search,
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
<<<<<<< HEAD


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
=======
>>>>>>> 529d0c047685b33cd9eeea4c9263603cd35fef91
