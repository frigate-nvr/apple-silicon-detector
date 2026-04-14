# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Unified analysis for the single helper binary
a = Analysis(
    ['detector/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['onnxruntime', 'cv2', 'zmq', 'pydantic', 'numpy', 'sympy', 'detector.service_manager', 'detector.zmq_onnx_client'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PIL', 'tkinter', 'PyQt5', 'IPython', 'jedi', 'setuptools', 'notebook'],
    win_no_prefer_redirects=False,
    noarchive=False,
)



pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Build the executable stub
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Collect everything into a directory (standard for macOS apps)
# This prevents the "extraction penalty" on every launch.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='detector',
)

