# translatorhoi4.spec
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
    d, b, h = collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)

for pkg in [
    "PyQt6",
    "g4f",
    "googletrans",
    "deep_translator",
    "gradio_client",
    "regex",
    "certifi",
    "httpx",
    "httpcore",
    "aiohttp",
    "idna",
    "chardet",
]:
    add_pkg(pkg)

hiddenimports += collect_submodules("PyQt6")

binaries += collect_dynamic_libs("curl_cffi", dependencies=True)

a = Analysis(
    [entry_script],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    console=False,
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
