"""DeepL translation backend."""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from .base import TranslationBackend


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
