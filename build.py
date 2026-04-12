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

# Packages to include (critical dependencies that are fully used)
INCLUDE_PACKAGES = [
    "translatorhoi4",
    "openai",
    "google",
    "deep_translator",
    "aiohttp",
    "requests",
    "qfluentwidgets",
    "jinja2",  # Prevent Nuitka inline copy conflict with pkg_resources
]

# Packages to follow imports (optional/lazily loaded - Nuitka includes only what's actually used)
FOLLOW_IMPORTS = [
    "googletrans",
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
        "--enable-plugin=pyside6",
        "--include-data-dir=assets=assets",
    ]

    # Include packages
    for pkg in INCLUDE_PACKAGES:
        cmd.append(f"--include-package={pkg}")

    # Follow imports (only include actually used modules from these packages)
    for pkg in FOLLOW_IMPORTS:
        cmd.append(f"--follow-import-to={pkg}")

    # Exclude modules
    for mod in EXCLUDE_MODULES:
        cmd.append(f"--nofollow-import-to={mod}")

    # Version info - only include if we have a valid numeric version
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

    # Parallel compilation (use all available CPU cores)
    cpu_count = os.cpu_count() or 1
    cmd.append(f"--jobs={cpu_count}")

    # macOS: Use clang directly as compiler (ccache as launcher via scons)
    if sys.platform == "darwin":
        cmd.extend(
            [
                "--clang",
                "--clang-executable=clang",
            ]
        )

    # Optimization
    cmd.extend(
        [
            "--assume-yes-for-downloads",
            "--remove-output",
        ]
    )

    # Platform-specific options
    if sys.platform == "win32":
        cmd.extend(
            [
                "--windows-icon-from-ico=assets/icon.png",
                "--windows-console-mode=disable",
                # Windows: disable LTO for faster builds (MSVC LTO is very slow)
                "--lto=no",
            ]
        )
    elif sys.platform == "darwin":
        cmd.extend(
            [
                "--macos-create-app-bundle",
                "--macos-app-icon=assets/icon.png",
                "--macos-app-name=TranslatorHoi4",
                # macOS: let Nuitka picks the best LTO for clang
                "--lto=auto",
            ]
        )
    elif sys.platform == "linux":
        cmd.extend(
            [
                "--linux-icon=assets/icon.png",
                # Linux: keep full LTO (gcc handles it well)
                "--lto=yes",
            ]
        )

    # Add the entry point
    # Use --module-name-base to ensure .dist folder is named after OUTPUT_NAME
    cmd.append(str(PROJECT_ROOT / "translatorhoi4" / "app.py"))

    return cmd


def find_nuitka_output():
    """Find the actual Nuitka output directory.

    Nuitka names the .dist folder after the entry point script (e.g. app.dist),
    not after --output-filename. This function searches for the correct path.
    """
    if not BUILD_DIR.exists():
        return None

    # Platform-specific expected outputs
    if sys.platform == "darwin":
        # macOS: look for .app bundle
        candidates = [
            BUILD_DIR / f"{OUTPUT_NAME}.app",
            BUILD_DIR / "app.app",
        ]
        # Also check inside any *.dist folder for .app
        for d in BUILD_DIR.iterdir():
            if d.is_dir() and d.name.endswith(".dist"):
                app_in_dist = d / f"{OUTPUT_NAME}.app"
                if app_in_dist.exists():
                    return app_in_dist
        for c in candidates:
            if c.exists():
                return c
    else:
        # Windows/Linux: look for .dist folder
        candidates = [
            BUILD_DIR / f"{OUTPUT_NAME}.dist",
            BUILD_DIR / "app.dist",
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c

    # Fallback: find any *.dist or *.app in BUILD_DIR
    for d in BUILD_DIR.iterdir():
        if d.is_dir() and (d.name.endswith(".dist") or d.name.endswith(".app")):
            return d

    return None


def clean_build_dirs():
    """Clean build and dist directories."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(exist_ok=True)

    # Clean old dist
    dist_path = DIST_DIR / OUTPUT_NAME
    if dist_path.exists():
        shutil.rmtree(dist_path)

    # Clean old macOS .app bundle if present
    app_path = DIST_DIR / f"{OUTPUT_NAME}.app"
    if app_path.exists():
        shutil.rmtree(app_path)


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
    # Fix Windows console encoding issues
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Building TranslatorHoi4 version: {APP_VERSION}")
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version}")

    check_macos_openssl()
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

    # Find actual Nuitka output (handles app.dist vs TranslatorHoi4.dist naming)
    output_path = find_nuitka_output()

    if output_path is None:
        print("ERROR: Build output not found!", file=sys.stderr)
        print(f"Searched in: {BUILD_DIR}", file=sys.stderr)
        if BUILD_DIR.exists():
            print(f"Build dir contents: {list(BUILD_DIR.iterdir())}", file=sys.stderr)
        sys.exit(1)

    # Determine final path based on platform
    if sys.platform == "darwin":
        final_path = DIST_DIR / f"{OUTPUT_NAME}.app"
    else:
        final_path = DIST_DIR / OUTPUT_NAME

    # Remove existing output if present
    if final_path.exists():
        shutil.rmtree(final_path)
    shutil.move(str(output_path), str(final_path))
    print(f"\n✓ Build successful!")
    print(f"Output: {final_path}")

    # Print size info
    total_size = sum(
        f.stat().st_size for f in final_path.rglob("*") if f.is_file()
    )
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
