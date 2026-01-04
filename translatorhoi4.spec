# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs

block_cipher = None
entry_script = "translatorhoi4/app.py"
app_name = "TranslatorHoi4"
icon_path = "assets/icon.png"

datas = []
binaries = []
hiddenimports = []

def add_pkg(pkg):
    """
    Helper to collect all resources, binaries and hidden imports for a package.
    """
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d)
        binaries.extend(b)
        hiddenimports.extend(h)
    except Exception as e:
        print(f"Warning: Could not collect '{pkg}': {e}")


packages_to_collect = [
    "qfluentwidgets",      # ВАЖНО: иконки и стили для нового GUI
    "g4f",                 # Free AI providers
    "anthropic",           # New API
    "google.generativeai", # Google Gemini API
    "googletrans",
    "deep_translator",
    "gradio_client",
    "openai",
    "requests",
    "regex",
    "certifi",
    "httpx",
    "httpcore",
    "aiohttp",
    "idna",
    "chardet",
    "curl_cffi",
    "yaml",
]

for pkg in packages_to_collect:
    add_pkg(pkg)


try:
    binaries += collect_dynamic_libs("curl_cffi")
except Exception:
    pass


if Path("assets/icon.png").exists():
    datas.append(("assets/icon.png", "assets"))

def _dedup_tuples(items):
    seen = set()
    out = []
    for it in items:
        # Кортежи (src, dst) делаем уникальными
        t_it = tuple(it) if isinstance(it, list) else it
        if t_it in seen:
            continue
        seen.add(t_it)
        out.append(it)
    return out

datas = _dedup_tuples(datas)
binaries = _dedup_tuples(binaries)
hiddenimports = sorted(set(hiddenimports))


hiddenimports += [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "qfluentwidgets.components",
    "qfluentwidgets.common",
]

a = Analysis(
    [entry_script],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["speech_recognition", "tkinter"], 
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    icon=icon_path if Path(icon_path).exists() else None,
    console=False, # False = без черного окна консоли
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False, 
    upx_exclude=[],
    name=app_name,
)