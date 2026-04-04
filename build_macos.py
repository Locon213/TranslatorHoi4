#!/usr/bin/env python3
"""
macOS build script using PyInstaller (Nuitka doesn't support PyQt6 on macOS).
"""
import sys
import os
import platform
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "build_pyinstaller"
DIST_DIR = PROJECT_ROOT / "dist"
OUTPUT_NAME = "TranslatorHoi4"
APP_VERSION = os.environ.get("APP_VERSION", "dev")


def get_pyinstaller_command():
    """Build the PyInstaller command for macOS."""
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=" + OUTPUT_NAME,
        "--onedir",
        "--windowed",
        "--icon=assets/icon.png",
        "--add-data=assets:assets",
        "--hidden-import=translatorhoi4",
        "--exclude-module=PyQt6.QtWebEngineCore",
        "--exclude-module=PyQt6.QtWebEngineWidgets",
        "--exclude-module=PyQt6.QtQml",
        "--exclude-module=PyQt6.QtQuick",
        "--exclude-module=PyQt6.Qt3DCore",
        "--exclude-module=PyQt6.Qt3DInput",
        "--exclude-module=PyQt6.Qt3DRender",
        "--exclude-module=PyQt6.QtMultimedia",
        "--exclude-module=PyQt6.QtSql",
        "--exclude-module=PyQt6.QtNetworkAuth",
        "--exclude-module=PyQt6.QtNfc",
        "--exclude-module=PyQt6.QtBluetooth",
        "--exclude-module=PyQt6.QtPositioning",
        "--exclude-module=PyQt6.QtSensors",
        "--exclude-module=PyQt6.QtSerialPort",
        "--exclude-module=PyQt6.QtCharts",
        "--exclude-module=PyQt6.QtDataVisualization",
        "--noconfirm",
        "--clean",
        "translatorhoi4/app.py",
    ]
    return cmd


def clean_build_dirs():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(exist_ok=True)


def main():
    print(f"Building TranslatorHoi4 (macOS/PyInstaller) version: {APP_VERSION}")
    print(f"Platform: {platform.system()} ({platform.machine()})")

    clean_build_dirs()

    env = os.environ.copy()
    env["APP_VERSION"] = APP_VERSION

    cmd = get_pyinstaller_command()
    print(f"\nRunning PyInstaller...")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)

    if result.returncode != 0:
        print("ERROR: Build failed!", file=sys.stderr)
        sys.exit(1)

    # PyInstaller puts output directly in dist/
    dist_path = DIST_DIR / OUTPUT_NAME
    if dist_path.exists():
        print(f"\n✓ Build successful!")
        print(f"Output directory: {dist_path}")

        total_size = sum(
            f.stat().st_size for f in dist_path.rglob("*") if f.is_file()
        )
        print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    else:
        print("ERROR: Build output not found!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
