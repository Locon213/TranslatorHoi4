from __future__ import annotations

from pathlib import Path

from translatorhoi4.translator.backends.base import TranslationBackend
from translatorhoi4.translator.config import JobConfig
from translatorhoi4.translator.engine import MODEL_REGISTRY, RetranslateWorker, TranslateWorker
from translatorhoi4.translator.prompts import batch_wrap_with_markers, parse_batch_response
from translatorhoi4.utils.fs import _build_prev_map, compute_output_path


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
        self.structured_payloads = []

    def translate_structured_batch(self, batch_payload: str, src_lang: str, dst_lang: str) -> str:
        self.structured_batch_calls += 1
        self.structured_payloads.append(batch_payload)
        return "not a valid batch response"


class PartialStructuredBackend(FakeStructuredBackend):
    def translate_structured_batch(self, batch_payload: str, src_lang: str, dst_lang: str) -> str:
        self.structured_batch_calls += 1
        self.structured_payloads.append(batch_payload)
        return "B001: Privet"

    def translate_many(self, texts, src_lang: str, dst_lang: str):
        self.translate_many_calls.append(list(texts))
        return ["Mir" for _ in texts]

    def translate_one(self, text: str, src_lang: str, dst_lang: str) -> str:
        return "Mir"


class RepairStructuredBackend(FakeStructuredBackend):
    def translate_structured_batch(self, batch_payload: str, src_lang: str, dst_lang: str) -> str:
        self.structured_batch_calls += 1
        self.structured_payloads.append(batch_payload)
        if self.structured_batch_calls == 1:
            return "B001: Privet"
        return "B002: Mir"

    def translate_many(self, texts, src_lang: str, dst_lang: str):
        self.translate_many_calls.append(list(texts))
        return ["fallback" for _ in texts]


class EchoStructuredBackend(FakeStructuredBackend):
    def translate_structured_batch(self, batch_payload: str, src_lang: str, dst_lang: str) -> str:
        self.structured_batch_calls += 1
        self.structured_payloads.append(batch_payload)
        return "\n".join(
            f"{line.split(':', 1)[0]}: translated-{index}"
            for index, line in enumerate(batch_payload.splitlines(), start=1)
            if line.startswith("B")
        )


class TranslatingBatchBackend(FakeBatchBackend):
    def translate_one(self, text: str, src_lang: str, dst_lang: str) -> str:
        self.translate_one_calls += 1
        return f"ru-{text}"

    def translate_many(self, texts, src_lang: str, dst_lang: str):
        self.translate_many_calls.append(list(texts))
        return [f"ru-{text}" for text in texts]


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
    assert backend.structured_batch_calls == 2
    assert len(backend.translate_many_calls) == 1
    assert worker._metrics["structured_batch_failures"] == 2
    assert worker._metrics["structured_batch_fallbacks"] == 2


def test_batch_mode_preserves_partial_structured_results(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    worker = TranslateWorker(cfg)
    backend = PartialStructuredBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "World"\n',
    ]

    out = worker._process_file_lines_batch(lines, backend, "test.yml", {})

    assert 'KEY1:0 "Privet"' in out[1]
    assert 'KEY2:0 "Mir"' in out[2]
    assert out[2] != lines[2]
    assert backend.structured_batch_calls >= 2


def test_batch_mode_repairs_missing_structured_keys_before_translate_many(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    worker = TranslateWorker(cfg)
    backend = RepairStructuredBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "World"\n',
    ]

    out = worker._process_file_lines_batch(lines, backend, "test.yml", {})

    assert 'KEY1:0 "Privet"' in out[1]
    assert 'KEY2:0 "Mir"' in out[2]
    assert backend.structured_batch_calls == 2
    assert backend.translate_many_calls == []
    assert "B001:" in backend.structured_payloads[0]
    assert "B002:" in backend.structured_payloads[1]


def test_batch_mode_uses_unique_request_keys_for_duplicate_loc_keys(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    worker = TranslateWorker(cfg)
    backend = EchoStructuredBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY1:0 "Again"\n',
    ]

    out = worker._process_file_lines_batch(lines, backend, "test.yml", {})

    assert "B001:" in backend.structured_payloads[0]
    assert "B002:" in backend.structured_payloads[0]
    assert 'KEY1:0 "translated-1"' in out[1]
    assert 'KEY1:0 "translated-4"' in out[2]


def test_batch_mode_uses_bulk_cache_operations(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    worker = TranslateWorker(cfg)
    worker._cache = TrackingCache()
    backend = TranslatingBatchBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "World"\n',
    ]

    worker._process_file_lines_batch(lines, backend, "test.yml", {})

    assert worker._cache.get_many_called is True
    assert worker._cache.set_many_called is True


def test_line_mode_deduplicates_uncached_texts(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=False, batch_size=8)
    worker = TranslateWorker(cfg)
    worker._cache = TrackingCache()
    backend = TranslatingBatchBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "Hello"\n',
        ' KEY3:0 "World"\n',
    ]

    out = worker._process_file_lines(lines, backend, "test.yml", {})

    assert backend.translate_many_calls == [["Hello", "World"]]
    assert 'KEY1:0 "ru-Hello"' in out[1]
    assert 'KEY2:0 "ru-Hello"' in out[2]
    assert 'KEY3:0 "ru-World"' in out[3]


def test_batch_mode_deduplicates_uncached_texts(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    worker = TranslateWorker(cfg)
    backend = EchoStructuredBackend()

    lines = [
        "l_english:\n",
        ' KEY1:0 "Hello"\n',
        ' KEY2:0 "Hello"\n',
    ]

    out = worker._process_file_lines_batch(lines, backend, "test.yml", {})

    assert backend.structured_batch_calls == 1
    assert "B001:" in backend.structured_payloads[0]
    assert "B002:" not in backend.structured_payloads[0]
    assert 'KEY1:0 "translated-1"' in out[1]
    assert 'KEY2:0 "translated-1"' in out[2]


def test_reuse_previous_loc_only_uses_marked_lines(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=False, batch_size=8)
    cfg.reuse_prev_loc = True
    cfg.mark_loc_flag = True
    worker = TranslateWorker(cfg)
    backend = TranslatingBatchBackend()
    prev_file = tmp_path / "prev.yml"
    prev_file.write_text(
        'l_russian:\n KEY1:0 "Old translated" #LOC!\n KEY2:0 "Unmarked old"\n',
        encoding="utf-8",
    )
    prev_map = _build_prev_map(str(prev_file))

    out = worker._process_file_lines(
        [
            "l_english:\n",
            ' KEY1:0 "Hello"\n',
            ' KEY2:0 "World"\n',
        ],
        backend,
        "test_l_english.yml",
        prev_map,
    )

    assert 'KEY1:0 "Old translated" #LOC!' in out[1]
    assert 'KEY2:0 "ru-World" #LOC!' in out[2]
    assert backend.translate_one_calls == 1
    assert backend.translate_many_calls == []


def test_batch_mode_reuses_previous_loc_only_uses_marked_lines(tmp_path):
    cfg = _make_cfg(tmp_path, batch_translation=True, chunk_size=8)
    cfg.reuse_prev_loc = True
    cfg.mark_loc_flag = True
    worker = TranslateWorker(cfg)
    backend = TranslatingBatchBackend()
    prev_file = tmp_path / "prev.yml"
    prev_file.write_text(
        'l_russian:\n KEY1:0 "Old translated" #LOC!\n KEY2:0 "Unmarked old"\n',
        encoding="utf-8",
    )
    prev_map = _build_prev_map(str(prev_file))

    out = worker._process_file_lines_batch(
        [
            "l_english:\n",
            ' KEY1:0 "Hello"\n',
            ' KEY2:0 "World"\n',
        ],
        backend,
        "test_l_english.yml",
        prev_map,
    )

    assert 'KEY1:0 "Old translated" #LOC!' in out[1]
    assert 'KEY2:0 "ru-World" #LOC!' in out[2]
    assert backend.translate_one_calls == 1
    assert backend.translate_many_calls == []


def test_reuse_previous_loc_reads_existing_output_even_with_skip_existing(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, batch_translation=False, batch_size=8)
    cfg.reuse_prev_loc = True
    cfg.prev_loc_dir = cfg.out_dir
    cfg.skip_existing = True
    cfg.rename_files = True
    cfg.mark_loc_flag = True
    source_file = tmp_path / "src" / "localisation" / "english" / "test_l_english.yml"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        'l_english:\n KEY1:0 "Hello"\n KEY2:0 "World"\n',
        encoding="utf-8",
    )
    out_path = compute_output_path(str(source_file), cfg)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        'l_russian:\n KEY1:0 "Old translated" #LOC!\n KEY2:0 "Unmarked old"\n',
        encoding="utf-8-sig",
    )
    backend = TranslatingBatchBackend()
    monkeypatch.setitem(MODEL_REGISTRY, "fake", lambda: backend)

    worker = TranslateWorker(cfg)
    worker.run()

    output = out_file.read_text(encoding="utf-8-sig")
    assert 'KEY1:0 "Old translated" #LOC!' in output
    assert 'KEY2:0 "ru-World" #LOC!' in output
    assert backend.translate_one_calls == 1
    assert backend.translate_many_calls == []


def test_reuse_previous_loc_uses_out_dir_when_previous_folder_is_blank(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, batch_translation=False, batch_size=8)
    cfg.reuse_prev_loc = True
    cfg.prev_loc_dir = None
    cfg.rename_files = True
    cfg.mark_loc_flag = True
    source_file = tmp_path / "src" / "localisation" / "english" / "test_l_english.yml"
    source_file.parent.mkdir(parents=True)
    source_file.write_text('l_english:\n KEY1:0 "Hello"\n', encoding="utf-8")
    out_path = compute_output_path(str(source_file), cfg)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text('l_russian:\n KEY1:0 "Old translated" #LOC!\n', encoding="utf-8-sig")
    backend = TranslatingBatchBackend()
    monkeypatch.setitem(MODEL_REGISTRY, "fake", lambda: backend)

    worker = TranslateWorker(cfg)
    worker.run()

    output = out_file.read_text(encoding="utf-8-sig")
    assert 'KEY1:0 "Old translated" #LOC!' in output
    assert backend.translate_one_calls == 0
    assert backend.translate_many_calls == []


def test_batch_prompt_and_parser_roundtrip_shape():
    payload = batch_wrap_with_markers({"KEY1": "Hello", "KEY2": "World"})
    parsed = parse_batch_response("KEY1: Privet\nKEY2: Mir", ["KEY1", "KEY2"])

    assert "<<SEG" in payload
    assert "<<END" in payload
    assert parsed == {"KEY1": "Privet", "KEY2": "Mir"}


def test_batch_parser_accepts_marker_block_response():
    response = """B001: <<SEG abc>>
Privet
<<END abc>>
B002: <<SEG def>>
Mir
<<END def>>"""

    parsed = parse_batch_response(response, ["B001", "B002"])

    assert parsed == {"B001": "Privet", "B002": "Mir"}


def test_batch_parser_accepts_full_localisation_lines():
    parsed = parse_batch_response('KEY1:0 "Privet" #LOC!\nKEY2:0 "Mir"', ["KEY1", "KEY2"])

    assert parsed == {"KEY1": "Privet", "KEY2": "Mir"}


def test_batch_parser_accepts_original_key_aliases_for_localisation_lines():
    parsed = parse_batch_response(
        'KEY1:0 "Privet" #LOC!\nKEY2:0 "Mir"',
        ["B001", "B002"],
        aliases={"KEY1": "B001", "KEY2": "B002"},
    )

    assert parsed == {"B001": "Privet", "B002": "Mir"}


def test_retranslate_worker_batches_uncached_items(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, batch_translation=False, batch_size=8)
    backend = TranslatingBatchBackend()
    monkeypatch.setitem(MODEL_REGISTRY, "fake", lambda: backend)
    worker = RetranslateWorker(
        cfg,
        [
            {"key": "KEY1", "original": "Hello", "row": 1},
            {"key": "KEY2", "original": "World", "row": 2},
            {"key": "KEY3", "original": "Hello", "row": 3},
        ],
    )
    emitted = []
    worker.translation_done.connect(lambda results: emitted.extend(results))

    worker.run()

    assert backend.translate_many_calls == [["Hello", "World"]]
    assert [item["translation"] for item in emitted] == ["ru-Hello", "ru-World", "ru-Hello"]
