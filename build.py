#!/usr/bin/env python3
"""
Build script for TranslatorHoi4 using Nuitka.
Replaces the project's old frozen-build flow.

Usage:
    python build.py              # Build with defaults
    APP_VERSION=1.5 python build.py  # Build with specific version
"""
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "build_nuitka"
DIST_DIR = PROJECT_ROOT / "dist"
OUTPUT_NAME = "TranslatorHoi4"
BUILD_META_FILE = PROJECT_ROOT / "translatorhoi4" / "_build_meta.py"

# Get version from environment or git
APP_VERSION = os.environ.get("APP_VERSION", "dev")

# Packages to include (critical dependencies that are fully used)
INCLUDE_PACKAGES = [
    "translatorhoi4",
    "openai",
    "google.genai",
    "googletrans",
    "deep_translator",
    "aiohttp",
    "requests",
    "qfluentwidgets",
    "jinja2",
]

# Packages to follow imports (optional/lazily loaded - Nuitka includes only what's actually used)
FOLLOW_IMPORTS = [
    "deepl",
    "groq",
    "together",
    "mistralai",
    "anthropic",
    "dotenv",
    "loguru",
    "toml",
    "psutil",
]

# Modules to exclude
EXCLUDE_MODULES = [
    "tkinter",
    "unittest",
    "google.genai.types",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DRender",
    "PySide6.QtMultimedia",
    "PySide6.QtSql",
    "PySide6.QtNetworkAuth",
    "PySide6.QtNfc",
    "PySide6.QtBluetooth",
    "PySide6.QtPositioning",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
]


def parse_version_tuple(version_str):
    """Convert version string to tuple for Nuitka --file-version."""
    parts = version_str.replace("v", "").split(".")
    numeric_parts = []
    for part in parts:
        try:
            numeric_parts.append(int(part))
        except ValueError:
            break
    if not numeric_parts:
        return None
    while len(numeric_parts) < 4:
        numeric_parts.append(0)
    return ".".join(str(x) for x in numeric_parts[:4])


def get_nuitka_command():
    """Build the Nuitka command based on platform."""
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--output-dir=" + str(BUILD_DIR),
        "--output-filename=" + OUTPUT_NAME,
        "--enable-plugin=pyside6",
        "--include-data-dir=assets=assets",
    ]

    for pkg in INCLUDE_PACKAGES:
        cmd.append(f"--include-package={pkg}")

    for pkg in FOLLOW_IMPORTS:
        cmd.append(f"--follow-import-to={pkg}")

    for mod in EXCLUDE_MODULES:
        cmd.append(f"--nofollow-import-to={mod}")

    file_version = parse_version_tuple(APP_VERSION)
    if file_version:
        cmd.extend(
            [
                f"--product-version={file_version}",
                f"--file-version={file_version}",
                "--company-name=Locon213",
                "--product-name=TranslatorHoi4",
                "--file-description=Cross-platform Paradox localisation translator with AI",
                "--copyright=MIT",
            ]
        )

    cpu_count = os.cpu_count() or 1
    cmd.append(f"--jobs={cpu_count}")

    if sys.platform == "darwin":
        cmd.append("--clang")

    cmd.extend(
        [
            "--assume-yes-for-downloads",
            "--remove-output",
        ]
    )

    if sys.platform == "win32":
        cmd.extend(
            [
                "--windows-icon-from-ico=assets/icon.png",
                "--windows-console-mode=disable",
                "--lto=no",
                "--disable-ccache",
            ]
        )
    elif sys.platform == "darwin":
        cmd.extend(
            [
                "--macos-create-app-bundle",
                "--macos-app-icon=assets/icon.png",
                "--macos-app-name=TranslatorHoi4",
                "--lto=auto",
            ]
        )
    elif sys.platform == "linux":
        cmd.extend(
            [
                "--linux-icon=assets/icon.png",
                "--lto=yes",
            ]
        )

    cmd.append(str(PROJECT_ROOT / "translatorhoi4" / "app.py"))
    return cmd


def find_nuitka_output():
    """Find the actual Nuitka output directory."""
    if not BUILD_DIR.exists():
        return None

    if sys.platform == "darwin":
        candidates = [
            BUILD_DIR / f"{OUTPUT_NAME}.app",
            BUILD_DIR / "app.app",
        ]
        for directory in BUILD_DIR.iterdir():
            if directory.is_dir() and directory.name.endswith(".dist"):
                app_in_dist = directory / f"{OUTPUT_NAME}.app"
                if app_in_dist.exists():
                    return app_in_dist
        for candidate in candidates:
            if candidate.exists():
                return candidate
    else:
        candidates = [
            BUILD_DIR / f"{OUTPUT_NAME}.dist",
            BUILD_DIR / "app.dist",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

    for directory in BUILD_DIR.iterdir():
        if directory.is_dir() and (directory.name.endswith(".dist") or directory.name.endswith(".app")):
            return directory

    return None


def clean_build_dirs():
    """Clean build and dist directories."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(exist_ok=True)

    dist_path = DIST_DIR / OUTPUT_NAME
    if dist_path.exists():
        shutil.rmtree(dist_path)

    app_path = DIST_DIR / f"{OUTPUT_NAME}.app"
    if app_path.exists():
        shutil.rmtree(app_path)


def _normalize_arch(machine: str) -> str:
    machine = machine.lower()
    if machine in {"amd64", "x86_64", "x64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return machine


def write_build_metadata() -> str | None:
    """Write temporary embedded build metadata for packaged runs."""
    previous = None
    if BUILD_META_FILE.exists():
        previous = BUILD_META_FILE.read_text(encoding="utf-8")

    content = textwrap.dedent(
        f"""\
        # Auto-generated by build.py
        BUILD_VERSION = {APP_VERSION!r}
        BUILD_CHANNEL = {"release" if APP_VERSION != "dev" else "dev"!r}
        BUILD_PLATFORM = {sys.platform!r}
        BUILD_ARCH = {_normalize_arch(platform.machine())!r}
        """
    )
    BUILD_META_FILE.write_text(content, encoding="utf-8")
    return previous


def restore_build_metadata(previous: str | None) -> None:
    """Restore build metadata file to its original state."""
    if previous is None:
        if BUILD_META_FILE.exists():
            BUILD_META_FILE.unlink()
        return
    BUILD_META_FILE.write_text(previous, encoding="utf-8")


def check_macos_openssl():
    if sys.platform != "darwin":
        return

    lib_paths = [
        Path("/usr/local/lib/libcrypto.dylib"),
        Path("/opt/homebrew/lib/libcrypto.dylib"),
    ]

    for libcrypto in lib_paths:
        if libcrypto.exists() and libcrypto.is_symlink():
            target = os.readlink(libcrypto)
            if "openssl@3" in str(target):
                print(f"WARNING: Nuitka cannot handle Homebrew openssl@3 symlink: {libcrypto} -> {target}")
                print("Attempting to unlink openssl@3...")
                try:
                    subprocess.run(["brew", "unlink", "openssl@3"], check=True)
                    print("Successfully unlinked openssl@3")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("ERROR: Could not automatically unlink openssl@3.")
                    print("Run manually: brew unlink openssl@3")
                    sys.exit(1)


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Building TranslatorHoi4 version: {APP_VERSION}")
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version}")

    check_macos_openssl()
    clean_build_dirs()
    previous_build_meta = write_build_metadata()

    try:
        cmd = get_nuitka_command()
        print("\nRunning Nuitka...")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            print("ERROR: Build failed!", file=sys.stderr)
            sys.exit(1)

        DIST_DIR.mkdir(exist_ok=True)
        output_path = find_nuitka_output()
        if output_path is None:
            print("ERROR: Build output not found!", file=sys.stderr)
            print(f"Searched in: {BUILD_DIR}", file=sys.stderr)
            if BUILD_DIR.exists():
                print(f"Build dir contents: {list(BUILD_DIR.iterdir())}", file=sys.stderr)
            sys.exit(1)

        final_path = DIST_DIR / (f"{OUTPUT_NAME}.app" if sys.platform == "darwin" else OUTPUT_NAME)
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.move(str(output_path), str(final_path))

        print("\nBuild successful!")
        print(f"Output: {final_path}")
        total_size = sum(file.stat().st_size for file in final_path.rglob("*") if file.is_file())
        print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    finally:
        restore_build_metadata(previous_build_meta)


if __name__ == "__main__":
    main()
