"""Application version and build metadata helpers."""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Dict

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

from .. import _build_meta


def _normalize_arch(machine: str | None = None) -> str:
    value = (machine or platform.machine() or "").lower()
    if value in {"amd64", "x86_64", "x64"}:
        return "x64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    return value or "unknown"


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists() or tomllib is None:
        return "dev"

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return "dev"
    return data.get("project", {}).get("version", "dev")


def get_version() -> str:
    """Get the effective application version for this runtime."""
    embedded = getattr(_build_meta, "BUILD_VERSION", "") or ""
    if embedded and embedded != "dev":
        return embedded

    env_version = os.environ.get("APP_VERSION")
    if env_version:
        return env_version

    package_version = _pyproject_version()
    return package_version if package_version and package_version != "0.0.0" else "dev"


def get_build_channel() -> str:
    channel = getattr(_build_meta, "BUILD_CHANNEL", "") or ""
    if channel:
        return channel
    return "release" if get_version() != "dev" else "dev"


def get_build_platform() -> str:
    return getattr(_build_meta, "BUILD_PLATFORM", "") or platform.system().lower()


def get_build_arch() -> str:
    embedded = getattr(_build_meta, "BUILD_ARCH", "") or ""
    return _normalize_arch(embedded)


def get_version_info() -> Dict[str, str]:
    """Return canonical packaged version metadata."""
    return {
        "version": get_version(),
        "channel": get_build_channel(),
        "platform": get_build_platform(),
        "arch": get_build_arch(),
    }


# Module-level variable for easy access
__version__ = get_version()
