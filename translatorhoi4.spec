# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_qt_plugins

hiddenimports = []
hiddenimports += collect_submodules('g4f')
hiddenimports += collect_submodules('googletrans')
hiddenimports += collect_submodules('curl_cffi')
hiddenimports += collect_submodules('browser_cookie3')
hiddenimports += collect_submodules('certifi')

datas = []
datas += collect_data_files('PyQt6')
datas += collect_data_files('g4f')
datas += collect_data_files('curl_cffi')
datas += collect_data_files('certifi')
datas += collect_data_files('browser_cookie3')
datas += collect_qt_plugins('PyQt6', 'platforms')

a = Analysis([
    'translatorhoi4/__main__.py',
],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='translatorhoi4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='translatorhoi4',
)
