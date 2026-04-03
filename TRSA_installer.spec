# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TRSA ComfyUI Installer."""
import os

block_cipher = None

# Build relative to spec file location
script_dir = os.path.dirname(os.path.abspath(SPEC))
files_dir = os.path.join(script_dir, "Files")

a = Analysis(
    [os.path.join(files_dir, "script_files", "installer_core.py")],
    pathex=[os.path.join(files_dir, "script_files")],
    binaries=[],
    datas=[
        (os.path.join(files_dir, "script_files", "installer_core_lang.py"), "."),
    ],
    hiddenimports=[
        "packaging",
        "packaging.version",
        "rich",
        "rich.panel",
        "rich.console",
        "rich.progress",
        "rich.prompt",
        "rich.table",
        "rich.text",
        "pygments",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "numpy",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="TRSA_installer",
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
