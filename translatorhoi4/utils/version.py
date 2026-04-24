"""Application version and build metadata helpers."""
from __future__ import annotations

import os
import platform
from typing import Dict

from .. import _build_meta


def _normalize_arch(machine: str | None = None) -> str:
    value = (machine or platform.machine() or "").lower()
    if value in {"amd64", "x86_64", "x64"}:
        return "x64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    return value or "unknown"


def _env_version() -> str:
    for name in ("APP_VERSION", "PACKAGE_VERSION", "TRANSLATORHOI4_VERSION"):
        value = os.environ.get(name, "").strip()
        if value:
            return value.removeprefix("v")
    return ""


def get_version() -> str:
    """Get the effective application version for this runtime."""
    embedded = getattr(_build_meta, "BUILD_VERSION", "") or ""
    if embedded and embedded != "dev":
        return embedded.removeprefix("v")

    env_version = _env_version()
    if env_version:
        return env_version

    return "dev"


def get_build_channel() -> str:
    channel = getattr(_build_meta, "BUILD_CHANNEL", "") or ""
    if channel and channel != "dev":
        return channel
    env_channel = os.environ.get("BUILD_CHANNEL", "").strip()
    if env_channel:
        return env_channel
    if _env_version() and _env_version() != "dev":
        return "release"
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
