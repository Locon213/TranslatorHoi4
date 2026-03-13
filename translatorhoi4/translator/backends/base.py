"""Base translation backend and helper functions."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import List, Optional


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


# --- Base Backend ---

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
