import sqlite3

import pytest

from translatorhoi4.utils.settings import (
    load_encrypted_preset,
    load_preset,
    load_settings,
    save_encrypted_preset,
    save_preset,
    save_settings,
)


def test_settings_use_sqlite_and_encrypt_secrets_when_keyring_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("translatorhoi4.utils.settings._set_keyring_secret", lambda key, value: False)
    monkeypatch.setattr("translatorhoi4.utils.settings._get_keyring_secret", lambda key: None)
    db_path = tmp_path / "settings.sqlite3"

    assert save_settings(
        {
            "src_lang": "english",
            "dst_lang": "russian",
            "openai_api_key": "sk-test-secret",
        },
        path=str(db_path),
    )

    loaded = load_settings(path=str(db_path))
    assert loaded["src_lang"] == "english"
    assert loaded["dst_lang"] == "russian"
    assert loaded["openai_api_key"] == "sk-test-secret"

    with sqlite3.connect(db_path) as db:
        keys = {row[0] for row in db.execute("SELECT key FROM settings")}
        raw_secret = db.execute("SELECT value FROM secrets WHERE key = 'openai_api_key'").fetchone()[0]
    assert "openai_api_key" not in keys
    assert "sk-test-secret" not in raw_secret


def test_plain_preset_is_sqlite_and_strips_secrets(tmp_path):
    preset_path = tmp_path / "work.th4preset"

    save_preset(
        str(preset_path),
        {
            "model": "OpenAI Compatible API",
            "openai_api_key": "sk-test-secret",
        },
    )

    loaded = load_preset(str(preset_path))
    assert loaded["model"] == "OpenAI Compatible API"
    assert "openai_api_key" not in loaded


def test_encrypted_preset_roundtrip_includes_secrets(tmp_path):
    pytest.importorskip("cryptography")
    preset_path = tmp_path / "work.th4preset.enc"

    save_encrypted_preset(
        str(preset_path),
        {
            "model": "OpenAI Compatible API",
            "openai_api_key": "sk-test-secret",
        },
        "correct horse battery staple",
    )

    loaded = load_encrypted_preset(str(preset_path), "correct horse battery staple")
    assert loaded["model"] == "OpenAI Compatible API"
    assert loaded["openai_api_key"] == "sk-test-secret"

    with pytest.raises(Exception):
        load_encrypted_preset(str(preset_path), "wrong password")
