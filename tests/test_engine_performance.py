from __future__ import annotations

from translatorhoi4.translator.backends.base import TranslationBackend
from translatorhoi4.translator.config import JobConfig
from translatorhoi4.translator.engine import TranslateWorker
from translatorhoi4.translator.prompts import batch_wrap_with_markers, parse_batch_response


def _make_cfg(tmp_path, *, batch_translation: bool = False, batch_size: int = 8, chunk_size: int = 8) -> JobConfig:
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    out_dir.mkdir()
    return JobConfig(
        src_dir=str(src_dir),
        out_dir=str(out_dir),
        src_lang="english",
        dst_lang="russian",
        model_key="fake",
        temperature=0.0,
        in_place=False,
        skip_existing=False,
        strip_md=False,
        batch_size=batch_size,
        rename_files=False,
        files_concurrency=1,
        key_skip_regex=None,
        cache_path=str(tmp_path / "cache.db"),
        batch_translation=batch_translation,
        chunk_size=chunk_size,
    )


class FakeBatchBackend(TranslationBackend):
    supports_batch = True

    def __init__(self):
        super().__init__(rpm_limit=0)
        self.translate_one_calls = 0
        self.translate_many_calls = []

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        return self.translate_one(text, src_lang, dst_lang)

    def translate_one(self, text: str, src_lang: str, dst_lang: str) -> str:
        self.translate_one_calls += 1
        return text

    def translate_many(self, texts, src_lang: str, dst_lang: str):
        self.translate_many_calls.append(list(texts))
        return list(texts)


class FakeStructuredBackend(FakeBatchBackend):
    supports_structured_batch = True

    def __init__(self):
        super().__init__()
        self.structured_batch_calls = 0

    def translate_structured_batch(self, batch_payload: str, src_lang: str, dst_lang: str) -> str:
        self.structured_batch_calls += 1
        return "not a valid batch response"


class TrackingCache:
    def __init__(self):
        self.values = {}
        self.get_many_called = False
        self.set_many_called = False

    def load(self):
        pass

    def save(self):
        pass

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def get_many(self, keys):
        self.get_many_called = True
        return {key: self.values[key] for key in keys if key in self.values}

    def set_many(self, entries):
        self.set_many_called = True
        self.values.update(entries)


def test_non_google_path_uses_translate_many(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=False, batch_size=8)
    worker = TranslateWorker(cfg)
    backend = FakeBatchBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "World"\n',
        ' KEY3:0 "Again"\n',
    ]

    out = worker._process_file_lines(lines, backend, "test.yml", {})

    assert out[0] == "l_russian:\n"
    assert len(backend.translate_many_calls) == 1
    assert backend.translate_one_calls == 0
    assert 'KEY1:0 "Hello"' in out[1]
    assert worker._metrics["cache_misses"] == 3


def test_batch_mode_falls_back_to_translate_many_on_structured_parse_failure(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    worker = TranslateWorker(cfg)
    backend = FakeStructuredBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "World"\n',
    ]

    out = worker._process_file_lines_batch(lines, backend, "test.yml", {})

    assert out[0] == "l_russian:\n"
    assert backend.structured_batch_calls == 1
    assert len(backend.translate_many_calls) == 1
    assert worker._metrics["structured_batch_failures"] == 1
    assert worker._metrics["structured_batch_fallbacks"] == 1


def test_batch_mode_uses_bulk_cache_operations(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    worker = TranslateWorker(cfg)
    worker._cache = TrackingCache()
    backend = FakeBatchBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "World"\n',
    ]

    worker._process_file_lines_batch(lines, backend, "test.yml", {})

    assert worker._cache.get_many_called is True
    assert worker._cache.set_many_called is True


def test_batch_prompt_and_parser_roundtrip_shape():
    payload = batch_wrap_with_markers({"KEY1": "Hello", "KEY2": "World"})
    parsed = parse_batch_response("KEY1: Privet\nKEY2: Mir", ["KEY1", "KEY2"])

    assert "<<SEG" in payload
    assert "<<END" in payload
    assert parsed == {"KEY1": "Privet", "KEY2": "Mir"}
