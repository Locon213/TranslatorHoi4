"""Google Free translation backend."""
from __future__ import annotations

from typing import List

from .base import TranslationBackend


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
