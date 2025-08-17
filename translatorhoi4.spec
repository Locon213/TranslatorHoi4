# translatorhoi4.spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs
from PyInstaller.utils.hooks.qt import collect_qt_plugins

block_cipher = None

entry_script = "translatorhoi4/app.py"
app_name = "TranslatorHoi4"
icon_path = "assets/icon.png"

hiddenimports = []
hiddenimports += collect_submodules("googletrans")
hiddenimports += collect_submodules("deep_translator")
hiddenimports += collect_submodules("g4f")
hiddenimports += collect_submodules("httpx")
hiddenimports += collect_submodules("httpcore")
hiddenimports += collect_submodules("aiohttp")
hiddenimports += collect_submodules("regex")

datas = []
datas += collect_data_files("gradio_client", include_py_files=False)
datas += collect_data_files("regex", include_py_files=False)
datas += collect_data_files("certifi", include_py_files=False)
datas += collect_qt_plugins(qt_plugins=["platforms", "styles", "imageformats", "iconengines", "tls"])

binaries = []
binaries += collect_dynamic_libs("PyQt6")
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
