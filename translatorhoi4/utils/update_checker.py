"""Auto-update discovery, download, and platform handoff via GitHub Releases."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .version import __version__, get_build_arch, get_build_platform

STATE_DIR = Path.home() / ".translatorhoi4"
UPDATES_DIR = STATE_DIR / "updates"
STATE_FILE = STATE_DIR / "update_state.json"
DEFAULT_CHECK_INTERVAL_SECONDS = 12 * 60 * 60


def _normalize_arch(value: str | None) -> str:
    machine = (value or "").lower()
    if machine in {"amd64", "x86_64", "x64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return machine or "unknown"


def _runtime_platform() -> str:
    platform_name = get_build_platform().lower()
    if platform_name.startswith("win"):
        return "windows"
    if platform_name.startswith("darwin") or platform_name.startswith("mac"):
        return "macos"
    if platform_name.startswith("linux"):
        return "linux"
    return platform_name


def _state_default() -> Dict[str, Any]:
    return {
        "last_checked_at": 0.0,
        "latest_version": None,
        "release": None,
    }


def _load_state() -> Dict[str, Any]:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return _state_default()


def _save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _parse_version(version: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.strip().lstrip("v").split("."))
    except Exception:
        return (0,)


@dataclass
class ReleaseAsset:
    name: str
    browser_download_url: str
    size: int = 0
    content_type: str = ""

    @property
    def lower_name(self) -> str:
        return self.name.lower()

    @property
    def extension(self) -> str:
        return Path(self.name).suffix.lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "browser_download_url": self.browser_download_url,
            "size": self.size,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReleaseAsset":
        return cls(
            name=data.get("name", ""),
            browser_download_url=data.get("browser_download_url", ""),
            size=int(data.get("size", 0) or 0),
            content_type=data.get("content_type", "") or "",
        )


@dataclass
class ReleaseInfo:
    tag_name: str
    name: str
    body: str
    html_url: str
    published_at: str
    assets: List[ReleaseAsset]
    prerelease: bool
    draft: bool

    @property
    def version(self) -> str:
        return self.tag_name[1:] if self.tag_name.startswith("v") else self.tag_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_name": self.tag_name,
            "name": self.name,
            "body": self.body,
            "html_url": self.html_url,
            "published_at": self.published_at,
            "assets": [asset.to_dict() for asset in self.assets],
            "prerelease": self.prerelease,
            "draft": self.draft,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReleaseInfo":
        return cls(
            tag_name=data.get("tag_name", ""),
            name=data.get("name", ""),
            body=data.get("body", ""),
            html_url=data.get("html_url", ""),
            published_at=data.get("published_at", ""),
            assets=[ReleaseAsset.from_dict(asset) for asset in data.get("assets", [])],
            prerelease=bool(data.get("prerelease", False)),
            draft=bool(data.get("draft", False)),
        )


class UpdateChecker:
    """Check for updates and resolve the best asset for the current runtime."""

    def __init__(
        self,
        owner: str = "Locon213",
        repo: str = "TranslatorHoi4",
        current_version: Optional[str] = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.current_version = current_version or __version__
        self.runtime_platform = _runtime_platform()
        self.runtime_arch = _normalize_arch(get_build_arch())
        self._api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        self._latest_release: Optional[ReleaseInfo] = None

    def _should_check(self, force: bool, interval_seconds: int) -> bool:
        if force:
            return True
        state = _load_state()
        last_checked = float(state.get("last_checked_at", 0) or 0)
        return (time.time() - last_checked) >= interval_seconds

    def _fetch_latest_release(self) -> Optional[ReleaseInfo]:
        response = requests.get(
            self._api_url,
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=15,
        )
        response.raise_for_status()
        releases = response.json()
        for release in releases:
            if release.get("draft", False):
                continue
            return ReleaseInfo(
                tag_name=release.get("tag_name", ""),
                name=release.get("name", ""),
                body=release.get("body", ""),
                html_url=release.get("html_url", ""),
                published_at=release.get("published_at", ""),
                assets=[
                    ReleaseAsset(
                        name=asset.get("name", ""),
                        browser_download_url=asset.get("browser_download_url", ""),
                        size=int(asset.get("size", 0) or 0),
                        content_type=asset.get("content_type", "") or "",
                    )
                    for asset in release.get("assets", [])
                ],
                prerelease=bool(release.get("prerelease", False)),
                draft=bool(release.get("draft", False)),
            )
        return None

    def _asset_matches_platform(self, asset: ReleaseAsset) -> bool:
        name = asset.lower_name
        if self.runtime_platform == "windows":
            if not name.endswith(".exe"):
                return False
            if self.runtime_arch == "arm64":
                return "setup_arm64" in name or "arm64" in name
            return "setup" in name and "arm64" not in name
        if self.runtime_platform == "macos":
            return name.endswith(".dmg") and self.runtime_arch in name
        if self.runtime_platform == "linux":
            if self.runtime_arch == "arm64":
                arch_markers = ("arm64", "aarch64")
            else:
                arch_markers = ("amd64", "x64", "x86_64")
            if not any(marker in name for marker in arch_markers):
                return False
            if self._preferred_linux_extension() == ".deb":
                return name.endswith(".deb")
            if self._preferred_linux_extension() == ".rpm":
                return name.endswith(".rpm")
            return name.endswith(".deb") or name.endswith(".rpm")
        return False

    def _preferred_linux_extension(self) -> str:
        if self.runtime_platform != "linux":
            return ""
        if shutil.which("dpkg"):
            return ".deb"
        if shutil.which("rpm"):
            return ".rpm"
        return ""

    def select_asset(self, release: ReleaseInfo) -> Optional[ReleaseAsset]:
        matched = [asset for asset in release.assets if self._asset_matches_platform(asset)]
        if matched:
            return matched[0]

        if self.runtime_platform == "linux":
            extension = ".deb" if self._preferred_linux_extension() == ".deb" else ".rpm"
            for asset in release.assets:
                if asset.extension == extension and self.runtime_arch in asset.lower_name:
                    return asset

        return release.assets[0] if release.assets else None

    def get_update_info(
        self,
        force: bool = False,
        interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
    ) -> Dict[str, Any]:
        state = _load_state()
        release = None
        checked_now = False

        if self._should_check(force, interval_seconds):
            checked_now = True
            try:
                release = self._fetch_latest_release()
                state["last_checked_at"] = time.time()
                state["release"] = release.to_dict() if release else None
                state["latest_version"] = release.version if release else None
                _save_state(state)
            except Exception as exc:
                release = ReleaseInfo.from_dict(state["release"]) if state.get("release") else None
                return self._format_result(release, checked_now=False, error=str(exc), state=state)
        else:
            release = ReleaseInfo.from_dict(state["release"]) if state.get("release") else None

        return self._format_result(release, checked_now=checked_now, error=None, state=state)

    def _format_result(
        self,
        release: Optional[ReleaseInfo],
        checked_now: bool,
        error: Optional[str],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        asset = self.select_asset(release) if release else None
        update_available = bool(release and _parse_version(self.current_version) < _parse_version(release.version))
        return {
            "update_available": update_available,
            "current_version": self.current_version,
            "latest_version": release.version if release else None,
            "release_url": release.html_url if release else None,
            "release_notes": release.body if release else None,
            "download_url": asset.browser_download_url if asset else None,
            "asset_name": asset.name if asset else None,
            "platform": self.runtime_platform,
            "arch": self.runtime_arch,
            "prerelease": release.prerelease if release else False,
            "checked_now": checked_now,
            "last_checked_at": state.get("last_checked_at", 0),
            "error": error,
            "assets": [item.to_dict() for item in (release.assets if release else [])],
        }


class UpdateDownloader:
    """Download and hand off updates to the platform installer UX."""

    def __init__(self, checker: Optional[UpdateChecker] = None) -> None:
        self.checker = checker or get_update_checker()
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def download_latest(self, force_check: bool = True) -> Tuple[Path, Dict[str, Any]]:
        info = self.checker.get_update_info(force=force_check)
        download_url = info.get("download_url")
        asset_name = info.get("asset_name")
        if not download_url or not asset_name:
            raise ValueError("No downloadable update asset is available for this platform")

        UPDATES_DIR.mkdir(parents=True, exist_ok=True)
        destination = UPDATES_DIR / asset_name

        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0) or 0)
        downloaded = 0

        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if self._progress_callback and total_size > 0:
                    self._progress_callback((downloaded / total_size) * 100.0, downloaded, total_size)

        return destination, info

    def open_installer(self, path: Path) -> str:
        path = path.resolve()
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return "Installer launched."
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return "Disk image opened in Finder."
        opener = shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, str(path)])
            return "Package opened with the system handler."
        return f"Downloaded update to {path}"

    def download_and_open_latest(self, force_check: bool = True) -> Dict[str, Any]:
        path, info = self.download_latest(force_check=force_check)
        message = self.open_installer(path)
        return {
            "path": str(path),
            "message": message,
            "asset_name": info.get("asset_name"),
            "latest_version": info.get("latest_version"),
            "release_url": info.get("release_url"),
        }


_update_checker: Optional[UpdateChecker] = None


def get_update_checker() -> UpdateChecker:
    global _update_checker
    if _update_checker is None:
        _update_checker = UpdateChecker()
    return _update_checker


def check_for_updates(force: bool = False, interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS) -> Dict[str, Any]:
    return get_update_checker().get_update_info(force=force, interval_seconds=interval_seconds)


def download_and_open_update(force_check: bool = True) -> Dict[str, Any]:
    downloader = UpdateDownloader(get_update_checker())
    return downloader.download_and_open_latest(force_check=force_check)
