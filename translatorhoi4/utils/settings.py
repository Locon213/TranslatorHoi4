"""Secure settings and preset persistence utilities."""
from __future__ import annotations

import base64
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import toml


APP_NAME = "TranslatorHoi4"
ORG_NAME = "Locon213"
KEYRING_SERVICE = "TranslatorHoi4"
SETTINGS_DB_FILE = "settings.sqlite3"
FALLBACK_SECRET_KEY_FILE = "settings.key"
LEGACY_SETTINGS_FILE = "translatorhoi4_settings.json"
PRESET_EXTENSION = ".th4preset"
ENCRYPTED_PRESET_EXTENSION = ".th4preset.enc"
SCHEMA_VERSION = 1


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered.endswith("_api_key")
        or lowered.endswith("_token")
        or lowered in {"api_key", "token", "password", "secret"}
    )


def _split_secrets(data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    regular: Dict[str, Any] = {}
    secrets: Dict[str, str] = {}
    for key, value in data.items():
        if _is_secret_key(key):
            if value not in (None, ""):
                secrets[key] = str(value)
            continue
        regular[key] = value
    return regular, secrets


def _get_keyring():
    try:
        import keyring  # type: ignore

        return keyring
    except Exception:
        return None


def _set_keyring_secret(key: str, value: str) -> bool:
    keyring = _get_keyring()
    if keyring is None:
        return False
    try:
        keyring.set_password(KEYRING_SERVICE, key, value)
        return True
    except Exception:
        return False


def _get_keyring_secret(key: str) -> Optional[str]:
    keyring = _get_keyring()
    if keyring is None:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, key)
    except Exception:
        return None


def _delete_secret(key: str) -> None:
    keyring = _get_keyring()
    if keyring is not None:
        try:
            keyring.delete_password(KEYRING_SERVICE, key)
        except Exception:
            pass


def get_config_dir() -> Path:
    """Return the per-user application config directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / ORG_NAME / APP_NAME

    try:
        from PySide6.QtCore import QStandardPaths

        qt_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        if qt_path:
            path = Path(qt_path)
            if path.name.lower() != APP_NAME.lower():
                path = path / APP_NAME
            return path
    except Exception:
        pass

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / ORG_NAME / APP_NAME
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / ORG_NAME / APP_NAME


def get_default_settings_path() -> str:
    """Get the default SQLite settings database path."""
    return str(get_config_dir() / SETTINGS_DB_FILE)


def _resolve_db_path(path: Optional[str] = None) -> Path:
    return Path(path or get_default_settings_path())


def get_legacy_settings_path() -> Path:
    return Path(__file__).resolve().parents[2] / LEGACY_SETTINGS_FILE


def _previous_config_dirs() -> list[Path]:
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_NAME,
            Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / APP_NAME,
        ]
    elif sys.platform == "darwin":
        candidates = [Path.home() / "Library" / "Application Support" / APP_NAME]
    else:
        candidates = [Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / APP_NAME]
    current = get_config_dir().resolve()
    return [path for path in candidates if path.resolve() != current]


def _migrate_previous_config_storage(path: Optional[str] = None) -> None:
    if path is not None:
        return

    target_db = Path(get_default_settings_path())
    if target_db.exists():
        return

    for old_dir in _previous_config_dirs():
        old_db = old_dir / SETTINGS_DB_FILE
        if not old_db.exists():
            continue

        target_db.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(str(old_db)) as source, sqlite3.connect(str(target_db)) as target:
                source.backup(target)
            old_key = old_dir / FALLBACK_SECRET_KEY_FILE
            new_key = _fallback_key_path()
            if old_key.exists() and not new_key.exists():
                shutil.copy2(old_key, new_key)
            return
        except Exception as exc:
            print(f"Failed to migrate previous settings database: {exc}")


def _connect(path: Optional[str] = None) -> sqlite3.Connection:
    db_path = _resolve_db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    _init_db(db)
    return db


def _init_db(db: sqlite3.Connection) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS settings (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL,
               updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS meta (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
           )"""
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS secrets (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL,
               backend TEXT NOT NULL,
               updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    db.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    db.commit()


def _write_settings_table(db: sqlite3.Connection, data: Dict[str, Any]) -> None:
    with db:
        db.execute("DELETE FROM settings")
        rows = [(key, json.dumps(value, ensure_ascii=False)) for key, value in data.items()]
        db.executemany(
            """INSERT INTO settings(key, value, updated_at)
               VALUES(?, ?, CURRENT_TIMESTAMP)""",
            rows,
        )


def _read_settings_table(db: sqlite3.Connection) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, raw in db.execute("SELECT key, value FROM settings"):
        try:
            data[key] = json.loads(raw)
        except Exception:
            data[key] = raw
    return data


def _fallback_key_path(db_path: Optional[str] = None) -> Path:
    if db_path:
        return Path(db_path).with_suffix(".key")
    return get_config_dir() / FALLBACK_SECRET_KEY_FILE


def _load_fallback_key(db_path: Optional[str] = None) -> bytes:
    key_path = _fallback_key_path(db_path)
    if key_path.exists():
        return base64.b64decode(key_path.read_text(encoding="ascii"))

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    key_path.write_text(base64.b64encode(key).decode("ascii"), encoding="ascii")
    try:
        os.chmod(key_path, 0o600)
    except Exception:
        pass
    return key


def _encrypt_fallback_secret(value: str, db_path: Optional[str] = None) -> str:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:
        raise RuntimeError("cryptography package is required for fallback secret storage") from exc

    nonce = os.urandom(12)
    ciphertext = AESGCM(_load_fallback_key(db_path)).encrypt(
        nonce,
        value.encode("utf-8"),
        b"TranslatorHoi4 local secret v1",
    )
    payload = {
        "version": 1,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "payload": base64.b64encode(ciphertext).decode("ascii"),
    }
    return json.dumps(payload, ensure_ascii=False)


def _decrypt_fallback_secret(raw: str, db_path: Optional[str] = None) -> Optional[str]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        payload = json.loads(raw)
        nonce = base64.b64decode(payload["nonce"])
        ciphertext = base64.b64decode(payload["payload"])
        plaintext = AESGCM(_load_fallback_key(db_path)).decrypt(
            nonce,
            ciphertext,
            b"TranslatorHoi4 local secret v1",
        )
        return plaintext.decode("utf-8")
    except Exception:
        return None


def _set_fallback_secret(db: sqlite3.Connection, key: str, value: str, db_path: Optional[str] = None) -> None:
    encrypted = _encrypt_fallback_secret(value, db_path)
    with db:
        db.execute(
            """INSERT OR REPLACE INTO secrets(key, value, backend, updated_at)
               VALUES(?, ?, 'encrypted-local', CURRENT_TIMESTAMP)""",
            (key, encrypted),
        )


def _get_fallback_secret(db: sqlite3.Connection, key: str, db_path: Optional[str] = None) -> Optional[str]:
    row = db.execute("SELECT value FROM secrets WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    return _decrypt_fallback_secret(row[0], db_path)


def _delete_fallback_secret(db: sqlite3.Connection, key: str) -> None:
    with db:
        db.execute("DELETE FROM secrets WHERE key = ?", (key,))


def _known_secret_keys(settings: Dict[str, Any]) -> set[str]:
    keys = {key for key in settings if _is_secret_key(key)}
    keys.update(
        {
            "g4f_api_key",
            "io_api_key",
            "openai_api_key",
            "anthropic_api_key",
            "gemini_api_key",
            "yandex_translate_api_key",
            "yandex_iam_token",
            "yandex_cloud_api_key",
            "deepl_api_key",
            "fireworks_api_key",
            "groq_api_key",
            "together_api_key",
            "mistral_api_key",
            "nvidia_api_key",
        }
    )
    return keys


def _migrate_legacy_json(db_path: Optional[str] = None) -> None:
    if db_path is not None:
        return

    legacy_path = get_legacy_settings_path()
    marker_path = get_config_dir() / ".legacy_settings_migrated"
    if marker_path.exists() or not legacy_path.exists():
        return

    try:
        with legacy_path.open("r", encoding="utf-8") as fh:
            legacy_data = json.load(fh)
        if not isinstance(legacy_data, dict):
            return

        save_settings(legacy_data, path=db_path)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("1", encoding="utf-8")
        backup = legacy_path.with_suffix(legacy_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(legacy_path, backup)
    except Exception as exc:
        print(f"Failed to migrate legacy settings: {exc}")


def save_settings(data: Dict[str, Any], path: Optional[str] = None) -> bool:
    """Persist settings to SQLite and store secret fields in the OS keyring."""
    try:
        regular, secrets = _split_secrets(dict(data))
        db_path = str(_resolve_db_path(path))
        with _connect(path) as db:
            _write_settings_table(db, regular)
            for key in _known_secret_keys(data):
                if key in secrets:
                    if _set_keyring_secret(key, secrets[key]):
                        _delete_fallback_secret(db, key)
                    else:
                        _set_fallback_secret(db, key, secrets[key], db_path)
                elif key in data and data.get(key) in (None, ""):
                    _delete_secret(key)
                    _delete_fallback_secret(db, key)
        return True
    except Exception as exc:
        print(f"Failed to save settings: {exc}")
        return False


def load_settings(path: Optional[str] = None) -> Dict[str, Any]:
    """Load settings from SQLite and hydrate secret fields from the OS keyring."""
    try:
        _migrate_previous_config_storage(path)
        _migrate_legacy_json(path)
        db_path = str(_resolve_db_path(path))
        with _connect(path) as db:
            data = _read_settings_table(db)
            for key in _known_secret_keys(data):
                secret = _get_keyring_secret(key) or _get_fallback_secret(db, key, db_path)
                if secret:
                    data[key] = secret
        return data
    except Exception as exc:
        print(f"Failed to load settings: {exc}")
        return {}


def _write_preset_sqlite(path: str, data: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        db = sqlite3.connect(temp_name)
        try:
            _init_db(db)
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('preset_format', 'translatorhoi4-preset')"
                )
                _write_settings_table(db, data)
        finally:
            db.close()
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _read_preset_sqlite(path: str) -> Dict[str, Any]:
    with sqlite3.connect(path) as db:
        _init_db(db)
        return _read_settings_table(db)


def save_preset(path: str, data: Dict[str, Any], *, include_secrets: bool = False) -> None:
    """Save a preset as a SQLite .th4preset file.

    Plain presets intentionally omit secrets. Use save_encrypted_preset when API
    keys must be exported.
    """
    preset_data = dict(data) if include_secrets else _split_secrets(dict(data))[0]
    _write_preset_sqlite(path, preset_data)


def load_preset(path: str) -> Dict[str, Any]:
    preset_path = Path(path)
    if preset_path.suffix.lower() == ".json":
        with preset_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    if str(preset_path).lower().endswith(ENCRYPTED_PRESET_EXTENSION):
        raise ValueError("Encrypted preset requires load_encrypted_preset()")
    return _read_preset_sqlite(str(preset_path))


def _derive_key(password: str, salt: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except Exception as exc:
        raise RuntimeError("cryptography package is required for encrypted presets") from exc

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    return kdf.derive(password.encode("utf-8"))


def save_encrypted_preset(path: str, data: Dict[str, Any], password: str) -> None:
    if not password:
        raise ValueError("Password is required")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:
        raise RuntimeError("cryptography package is required for encrypted presets") from exc

    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = _derive_key(password, salt)
    plaintext = toml.dumps(dict(data)).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, b"TranslatorHoi4 preset v1")
    envelope = {
        "format": "TranslatorHoi4 encrypted preset",
        "version": 1,
        "kdf": "PBKDF2-HMAC-SHA256",
        "iterations": 390000,
        "cipher": "AES-256-GCM",
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "payload": base64.b64encode(ciphertext).decode("ascii"),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(toml.dumps(envelope), encoding="utf-8")


def load_encrypted_preset(path: str, password: str) -> Dict[str, Any]:
    if not password:
        raise ValueError("Password is required")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:
        raise RuntimeError("cryptography package is required for encrypted presets") from exc

    envelope = toml.load(path)
    salt = base64.b64decode(envelope["salt"])
    nonce = base64.b64decode(envelope["nonce"])
    ciphertext = base64.b64decode(envelope["payload"])
    key = _derive_key(password, salt)
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, b"TranslatorHoi4 preset v1")
    data = toml.loads(plaintext.decode("utf-8"))
    return data if isinstance(data, dict) else {}
