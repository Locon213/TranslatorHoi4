# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for macOS builds

import os
import sys
import platform

block_cipher = None

# Detect target architecture
if 'ARCHFLAGS' in os.environ:
    # CI may set ARCHFLAGS
    arch_env = os.environ.get('ARCHFLAGS', '')
    target_arch = 'arm64' if 'arm64' in arch_env else 'x86_64'
elif platform.machine() == 'arm64':
    target_arch = 'arm64'
else:
    target_arch = 'x86_64'

# Include assets
datas = [
    ('assets', 'assets'),
]

excluded_modules = [
    'tkinter', 'unittest',
    'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.Qt3DCore', 'PyQt6.Qt3DInput',
    'PyQt6.Qt3DRender', 'PyQt6.QtMultimedia', 'PyQt6.QtSql',
    'PyQt6.QtNetworkAuth', 'PyQt6.QtNfc', 'PyQt6.QtBluetooth',
    'PyQt6.QtPositioning', 'PyQt6.QtSensors', 'PyQt6.QtSerialPort',
    'PyQt6.QtCharts', 'PyQt6.QtDataVisualization',
    # Exclude test/dev packages that cause runtime hook conflicts
    'pytest', 'py',
    # Exclude heavy packages not needed
    'scipy', 'numpy',
]

a = Analysis(
    ['translatorhoi4/app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'translatorhoi4',
        'qfluentwidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],  # No runtime hooks to avoid pkg_resources issues
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TranslatorHoi4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='TranslatorHoi4',
)

app = BUNDLE(
    coll,
    name='TranslatorHoi4.app',
    icon='assets/icon.png',
    bundle_identifier='com.locon213.translatorhoi4',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
    },
)
