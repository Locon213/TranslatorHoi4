"""Auto-update functionality via GitHub Releases."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReleaseInfo:
    """Information about a GitHub release."""
    tag_name: str
    name: str
    body: str
    html_url: str
    published_at: str
    assets: List[Dict[str, Any]]
    prerelease: bool
    draft: bool
    
    @property
    def version(self) -> str:
        """Extract version from tag name."""
        tag = self.tag_name
        if tag.startswith("v"):
            tag = tag[1:]
        return tag
    
    @property
    def download_url(self) -> Optional[str]:
        """Get the first asset download URL."""
        if not self.assets:
            return None
        # Prefer platform-specific assets
        platform = sys.platform
        for asset in self.assets:
            name = asset.get("name", "").lower()
            if "win" in platform and name.endswith(".exe"):
                return asset.get("browser_download_url")
            if "linux" in platform and name.endswith(".AppImage"):
                return asset.get("browser_download_url")
            if "darwin" in platform and name.endswith(".dmg"):
                return asset.get("browser_download_url")
        # Fall back to first asset
        return self.assets[0].get("browser_download_url") if self.assets else None


class UpdateChecker:
    """Check for updates from GitHub Releases."""
    
    def __init__(
        self,
        owner: str = "Locon213",
        repo: str = "TranslatorHoi4",
        current_version: Optional[str] = None,
    ):
        """Initialize the update checker.
        
        Args:
            owner: GitHub repository owner
            repo: GitHub repository name
            current_version: Current application version
        """
        self.owner = owner
        self.repo = repo
        self.current_version = current_version or self._get_current_version()
        self._api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        self._latest_release: Optional[ReleaseInfo] = None
        self._cached_version: Optional[str] = None
    
    def _get_current_version(self) -> str:
        """Get the current application version."""
        # Try to import from version module
        try:
            from ..utils.version import __version__
            return __version__
        except ImportError:
            pass
        
        # Try pyproject.toml
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject.exists():
            try:
                import toml
                with open(pyproject, "r", encoding="utf-8") as f:
                    data = toml.load(f)
                    return data.get("project", {}).get("version", "1.0.0")
            except Exception:
                pass
        
        # Check environment variable
        version = os.environ.get("APP_VERSION")
        if version:
            return version
        
        return "1.0.0"
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse version string to tuple."""
        try:
            parts = version.strip().lstrip("v").split(".")
            return tuple(int(p) for p in parts[:3])
        except (ValueError, AttributeError):
            return (0, 0, 0)
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two versions.
        
        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        p1 = self._parse_version(v1)
        p2 = self._parse_version(v2)
        
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
        return 0
    
    def check_for_updates(self, use_cache: bool = True) -> Tuple[bool, Optional[ReleaseInfo]]:
        """Check for available updates.
        
        Args:
            use_cache: Use cached result if available
            
        Returns:
            Tuple of (update_available, release_info)
        """
        if use_cache and self._cached_version:
            comparison = self._compare_versions(self.current_version, self._cached_version)
            return comparison < 0, self._latest_release
        
        try:
            import requests
            response = requests.get(
                self._api_url,
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            response.raise_for_status()
            
            releases = response.json()
            
            # Find the latest non-draft release
            for release in releases:
                if release.get("draft", False):
                    continue
                
                self._latest_release = ReleaseInfo(
                    tag_name=release.get("tag_name", ""),
                    name=release.get("name", ""),
                    body=release.get("body", ""),
                    html_url=release.get("html_url", ""),
                    published_at=release.get("published_at", ""),
                    assets=release.get("assets", []),
                    prerelease=release.get("prerelease", False),
                    draft=release.get("draft", False),
                )
                self._cached_version = self._latest_release.version
                
                comparison = self._compare_versions(self.current_version, self._cached_version)
                return comparison < 0, self._latest_release
            
            return False, None
            
        except Exception:
            return False, None
    
    @property
    def latest_release(self) -> Optional[ReleaseInfo]:
        """Get the latest release info."""
        if not self._latest_release:
            self.check_for_updates(use_cache=False)
        return self._latest_release
    
    def get_update_info(self) -> Dict[str, Any]:
        """Get detailed update information.
        
        Returns:
            Dictionary with update information
        """
        update_available, release = self.check_for_updates()
        
        if not update_available or not release:
            return {
                "update_available": False,
                "current_version": self.current_version,
                "latest_version": None,
                "release_url": None,
                "release_notes": None,
                "download_url": None,
                "prerelease": False,
            }
        
        return {
            "update_available": True,
            "current_version": self.current_version,
            "latest_version": release.version,
            "release_url": release.html_url,
            "release_notes": release.body,
            "download_url": release.download_url,
            "prerelease": release.prerelease,
            "assets": release.assets,
        }


class UpdateDownloader:
    """Download and apply updates."""
    
    def __init__(self, release: ReleaseInfo):
        """Initialize the downloader.
        
        Args:
            release: Release to download
        """
        self.release = release
        self._download_path: Optional[Path] = None
        self._progress_callback: Optional[callable] = None
    
    def set_progress_callback(self, callback: callable):
        """Set progress callback.
        
        Args:
            callback: Function to call with progress (percent, downloaded, total)
        """
        self._progress_callback = callback
    
    def download(self, output_dir: Optional[Path] = None) -> Path:
        """Download the release asset.
        
        Args:
            output_dir: Directory to save the download
            
        Returns:
            Path to downloaded file
        """
        import requests
        
        download_url = self.release.download_url
        if not download_url:
            raise ValueError("No download URL available for this release")
        
        if output_dir is None:
            output_dir = Path.home() / ".translatorhoi4" / "updates"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        asset_name = download_url.split("/")[-1].split("?")[0]
        self._download_path = output_dir / asset_name
        
        # Download with progress
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        
        with open(self._download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if self._progress_callback and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        self._progress_callback(percent, downloaded, total_size)
        
        return self._download_path
    
    def verify_checksum(self, expected_checksum: Optional[str] = None) -> bool:
        """Verify the downloaded file checksum.
        
        Args:
            expected_checksum: Expected SHA256 checksum
            
        Returns:
            True if checksum matches
        """
        if not self._download_path:
            return False
        
        import hashlib
        
        sha256_hash = hashlib.sha256()
        with open(self._download_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_checksum = sha256_hash.hexdigest()
        
        if expected_checksum:
            return actual_checksum.lower() == expected_checksum.lower()
        
        # If no expected checksum, just return True (download was successful)
        return True
    
    def extract_and_apply(self, backup_dir: Optional[Path] = None) -> bool:
        """Extract and apply the update.
        
        Args:
            backup_dir: Directory to backup current installation
            
        Returns:
            True if update was applied successfully
        """
        if not self._download_path:
            return False
        
        try:
            import zipfile
            import shutil
            import tempfile
            
            app_dir = Path(__file__).parent.parent.parent
            
            # Create backup if requested
            if backup_dir:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    app_dir,
                    backup_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    ignore=shutil.ignore_patterns("*.log", "logs/*", ".translatorhoi4*"),
                )
            
            # Extract to temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                if self._download_path.suffix == ".zip":
                    with zipfile.ZipFile(self._download_path, "r") as zip_ref:
                        zip_ref.extractall(temp_path)
                else:
                    # Assume it's a single file (like .AppImage)
                    shutil.copy(self._download_path, temp_path / self._download_path.name)
                
                # Find the extracted content
                extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
                if extracted_dirs:
                    source_dir = extracted_dirs[0]
                else:
                    source_dir = temp_path
                
                # Copy files to app directory
                for item in source_dir.iterdir():
                    if item.is_file():
                        dest = app_dir / item.name
                        shutil.copy2(item, dest)
                    elif item.is_dir():
                        dest = app_dir / item.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
            
            return True
            
        except Exception:
            return False
    
    @property
    def download_path(self) -> Optional[Path]:
        """Get the path to the downloaded file."""
        return self._download_path


# Global update checker instance
_update_checker: Optional[UpdateChecker] = None


def get_update_checker() -> UpdateChecker:
    """Get the global update checker instance."""
    global _update_checker
    if _update_checker is None:
        _update_checker = UpdateChecker()
    return _update_checker


def check_for_updates() -> Dict[str, Any]:
    """Check for available updates.
    
    Returns:
        Dictionary with update information
    """
    return get_update_checker().get_update_info()


# Import datetime for the backup function
from datetime import datetime
