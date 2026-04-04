#!/usr/bin/env python3
"""
Build script for TranslatorHoi4 using Nuitka.
Replaces the old PyInstaller .spec file.

Usage:
    python build.py              # Build with defaults
    APP_VERSION=1.5 python build.py  # Build with specific version
"""
import sys
import os
import platform
import shutil
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "build_nuitka"
DIST_DIR = PROJECT_ROOT / "dist"
OUTPUT_NAME = "TranslatorHoi4"

# Get version from environment or git
APP_VERSION = os.environ.get("APP_VERSION", "dev")

# Packages to include
INCLUDE_PACKAGES = [
    "translatorhoi4",
    "openai",
    "anthropic",
    "google",
    "googletrans",
    "deep_translator",
    "deepl",
    "groq",
    "together",
    "mistralai",
    "aiohttp",
    "requests",
    "regex",
    "yaml",
    "dotenv",
    "loguru",
    "toml",
    "fluentqt",
]

# Modules to exclude
EXCLUDE_MODULES = [
    "tkinter",
    "unittest",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtQml",
    "PyQt6.QtQuick",
    "PyQt6.Qt3DCore",
    "PyQt6.Qt3DInput",
    "PyQt6.Qt3DRender",
    "PyQt6.QtMultimedia",
    "PyQt6.QtSql",
    "PyQt6.QtNetworkAuth",
    "PyQt6.QtNfc",
    "PyQt6.QtBluetooth",
    "PyQt6.QtPositioning",
    "PyQt6.QtSensors",
    "PyQt6.QtSerialPort",
    "PyQt6.QtCharts",
    "PyQt6.QtDataVisualization",
]


def parse_version_tuple(version_str):
    """Convert version string to tuple for Nuitka --file-version.
    
    Nuitka requires file-version to be a numeric tuple like 1.5.0.0
    """
    parts = version_str.replace("v", "").split(".")
    numeric_parts = []
    for p in parts:
        try:
            numeric_parts.append(int(p))
        except ValueError:
            break
    if not numeric_parts:
        return None  # Invalid version like "dev"
    # Pad to max 4 parts
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
        "--enable-plugin=pyqt6",
        "--include-data-dir=assets=assets",
    ]

    # Include packages
    for pkg in INCLUDE_PACKAGES:
        cmd.append(f"--include-package={pkg}")

    # Exclude modules
    for mod in EXCLUDE_MODULES:
        cmd.append(f"--nofollow-import-to={mod}")

    # Version info
    file_version = parse_version_tuple(APP_VERSION)
    version_args = [
        f"--product-version={APP_VERSION}",
        "--company-name=Locon213",
        "--product-name=TranslatorHoi4",
        "--file-description=Cross-platform Paradox localisation translator with AI",
        "--copyright=MIT",
    ]
    if file_version:
        version_args.insert(1, f"--file-version={file_version}")

    cmd.extend(version_args)

    # Optimization
    cmd.extend(
        [
            "--assume-yes-for-downloads",
            "--remove-output",
            "--lto=yes",
            "--noinclude-default-mode=error",
        ]
    )

    # Platform-specific options
    if sys.platform == "win32":
        cmd.extend(
            [
                "--windows-icon-from-ico=assets/icon.png",
                "--windows-console-mode=disable",
            ]
        )
    elif sys.platform == "darwin":
        cmd.extend(
            [
                "--macos-create-app-bundle",
                "--macos-app-icon=assets/icon.png",
                "--macos-app-name=TranslatorHoi4",
            ]
        )
    elif sys.platform == "linux":
        cmd.extend(
            [
                "--linux-icon=assets/icon.png",
            ]
        )

    # Add the entry point
    cmd.append(str(PROJECT_ROOT / "translatorhoi4" / "app.py"))

    return cmd


def clean_build_dirs():
    """Clean build and dist directories."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(exist_ok=True)

    # Clean old dist
    dist_path = DIST_DIR / OUTPUT_NAME
    if dist_path.exists():
        shutil.rmtree(dist_path)


def main():
    print(f"Building TranslatorHoi4 version: {APP_VERSION}")
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version}")

    clean_build_dirs()

    cmd = get_nuitka_command()
    print(f"\nRunning Nuitka...")

    # Execute Nuitka with proper argument handling
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("ERROR: Build failed!", file=sys.stderr)
        sys.exit(1)

    # Move output to dist directory
    DIST_DIR.mkdir(exist_ok=True)
    output_path = BUILD_DIR / f"{OUTPUT_NAME}.dist"

    if output_path.exists():
        final_path = DIST_DIR / OUTPUT_NAME
        shutil.move(str(output_path), str(final_path))
        print(f"\n✓ Build successful!")
        print(f"Output directory: {final_path}")

        # Print size info
        total_size = sum(
            f.stat().st_size for f in final_path.rglob("*") if f.is_file()
        )
        print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    else:
        print("ERROR: Build output not found!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
