# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all


datas, binaries, hiddenimports = collect_all('qfluentwidgets')

# Include assets
datas += [('assets', 'assets')]

excluded_modules = [
    'tkinter',
    'unittest',
    'PyQt6.QtWebEngine',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebEngineWidgets',      
    'PyQt6.QtQml',
    'PyQt6.QtQuick',
    'PyQt6.Qt3DCore',
    'PyQt6.Qt3DInput',
    'PyQt6.Qt3DRender',
    'PyQt6.QtMultimedia',
    'PyQt6.QtSql',
    'PyQt6.QtNetworkAuth',
    'PyQt6.QtNfc',
    'PyQt6.QtBluetooth',
    'PyQt6.QtPositioning',
    'PyQt6.QtSensors',
    'PyQt6.QtSerialPort',
    'PyQt6.QtCharts',
    'PyQt6.QtDataVisualization'
]

# 3. Список файлов, которые НЕЛЬЗЯ сжимать UPX
upx_excludes = [
    'vcruntime140.dll',
    'python3.dll',
    'python310.dll',
    'python311.dll',
    'python312.dll',
    'Qt6Core.dll',
    'Qt6Gui.dll',
    'Qt6Widgets.dll',
    'qwindows.dll'
]

block_cipher = None

a = Analysis(
    ['translatorhoi4/app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    
    # --- UPX ---
    upx=True,
    upx_exclude=upx_excludes,
    # -----------
    
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
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
    
    # --- UPX ---
    upx=True,
    upx_exclude=upx_excludes,
    # -----------
    
    name='TranslatorHoi4',
)
