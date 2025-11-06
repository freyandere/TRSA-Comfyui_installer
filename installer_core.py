#!/usr/bin/env python3
# -*- coding: utf‑8 -*-

"""
ComfyUI – SageAttention installer

Features
--------
1. Friendly greeting screen (name + version centered)
2. Language selection – RU / EN (default = system locale)
3. Detect installed SageAttention, Torch and CUDA
4. Install the wheel that matches the current torch/cuda combo
5. Offer an upgrade if the bundle is older than 2.7.1+cu128
6. Skip reinstall when running the newest supported combo
   (torch = 2.9.0+cu130, cuda = 13.0)
7. Summary table – success / failure per step
8. Cleanup temporary wheel file

Author:  Your Name
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
import subprocess
import urllib.request
import zipfile
import tempfile
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# --------------------------------------------------------------------------- #
# 1. Optional Rich support – fall back to plain prints if not available      #
# --------------------------------------------------------------------------- #
try:
    from rich.console import Console
    from rich.table import Table
except Exception:          # pragma: no cover
    Console = None
    Table = None

console = Console() if Console else None


def msg(text: str, *, end="\n"):
    """Print message – Rich if available."""
    if console:
        console.print(text)
    else:
        print(text, end=end)


# --------------------------------------------------------------------------- #
# 2. Language handling                                                      #
# --------------------------------------------------------------------------- #
def _system_language() -> str:
    """
    Detect system language (ru/ en) with fallback to English.
    Checks env ACC_LANG first, then OS locale.
    """
    # Explicit override
    env_lang = os.getenv("ACC_LANG", "").strip().lower()
    if env_lang in ("ru", "en"):
        return env_lang

    # Environment variables that may contain the locale
    loc = (os.getenv("LANG") or os.getenv("LC_ALL") or "").lower()
    if loc.startswith(("ru", "ru_")):
        return "ru"
    # Default to English
    return "en"


def _prompt_language() -> str:
    """
    Ask user for language only if they wish.
    If the input is empty – use system default.
    """
    sys_lang = _system_language()
    msg("Select language / Выберите язык:")
    msg("  1) RU (Русский)")
    msg("  2) EN (English)")
    msg(f"Press Enter to keep system language ({sys_lang.upper()}).")
    choice = input("Choice (1/2, Enter=Auto): ").strip()
    if choice == "1":
        return "ru"
    if choice == "2":
        return "en"
    # Empty → system default
    return sys_lang


LANG = _prompt_language()

# --------------------------------------------------------------------------- #
# 3. Greeting screen                                                       #
# --------------------------------------------------------------------------- #
SCRIPT_NAME = Path(__file__).stem.replace("_", " ").title()
VERSION = "1.0.0"

def _greeting():
    """
    Prints a centered title (name + version) in the terminal window.
    """
    term_width = shutil.get_terminal_size((80, 20)).columns
    title = f"{SCRIPT_NAME} v{VERSION}"
    msg(title.center(term_width))
    msg("-" * len(title).center(term_width))

_greeting()


# --------------------------------------------------------------------------- #
# 4. Helpers – version parsing & comparison                                 #
# --------------------------------------------------------------------------- #
def _parse_version(v: str) -> Tuple[int, int, int]:
    """Return (major, minor, patch) from a dotted string."""
    parts = v.split(".")
    return tuple(int(p) for p in parts[:3])


def _ver_lt(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    return a < b


def _ver_ge(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    return a >= b


# --------------------------------------------------------------------------- #
# 5. Installed packages detection                                           #
# --------------------------------------------------------------------------- #
def _installed_version(pkg_name: str) -> Optional[str]:
    """Return package version or None if not installed."""
    try:
        # Python ≥3.8
        from importlib.metadata import version, PackageNotFoundError

        return version(pkg_name)
    except Exception:
        # Fallback – pip show
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "pip", "show", pkg_name],
                stderr=subprocess.DEVNULL,
            )
            for line in out.decode().splitlines():
                if line.startswith("Version:"):
                    return line.split(":")[1].strip()
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# 6. Torch & CUDA detection                                               #
# --------------------------------------------------------------------------- #
def _torch_and_cuda() -> Tuple[Optional[str], Optional[str]]:
    """Return (torch_version, cuda_version) or (None, None) if torch not found."""
    try:
        import torch
        tv = torch.__version__          # e.g. 2.9.0+cu130
        cv = torch.version.cuda        # e.g. "13.0"
        return tv, cv
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# 7. Wheel mapping – supported combos                                       #
# --------------------------------------------------------------------------- #
REPO_BASE = (
    "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main"
)

WHEEL_MAP: dict[Tuple[str, str], str] = {
    ("2.7.1", "12.8"): "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl",
    ("2.8.0", "12.8"): "sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl",
}
# Latest wheel – used for any torch >= 2.9.0 and CUDA >= 13.0
LATEST_WHEEL = (
    "sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
)


def _select_wheel(tver: str, cver: str) -> Optional[str]:
    """
    Return wheel filename that matches the torch/cuda combo.
    If no exact match – return None (upgrade needed).
    """
    # Exact combos
    key = (tver, cver)
    if key in WHEEL_MAP:
        return WHEEL_MAP[key]

    # Latest wheel for newer versions
    t_tuple = _parse_version(tver.split("+")[0])  # strip +cuXX
    if _ver_ge(t_tuple, (2, 9, 0)) and cver.startswith("13"):
        return LATEST_WHEEL

    # No match
    return None


# --------------------------------------------------------------------------- #
# 8. Download wheel                                                        #
# --------------------------------------------------------------------------- #
def _download_wheel(wheel_name: str, dest_dir: Path) -> Path:
    url = f"{REPO_BASE}/{wheel_name}"
    dest_path = dest_dir / wheel_name

    msg(f"Downloading {wheel_name} ...")
    with urllib.request.urlopen(url) as resp, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)
    return dest_path


# --------------------------------------------------------------------------- #
# 9. Install wheel via pip                                                 #
# --------------------------------------------------------------------------- #
def _pip_install(file_path: Path) -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", str(file_path)]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        success = result.returncode == 0
        return success, result.stdout
    except Exception as e:
        return False, str(e)


# --------------------------------------------------------------------------- #
# 10. Summary table                                                        #
# --------------------------------------------------------------------------- #
def _summary(results: list[Tuple[str, bool, Optional[str]]]):
    """Print a summary table – Rich if available."""
    if Table and console:
        tbl = Table(title="Installation Summary")
        tbl.add_column("Step", style="cyan")
        tbl.add_column("Status", style="magenta")
        for name, ok, msg_ in results:
            status = f"[green]OK[/green]" if ok else f"[red]FAIL[/red]"
            tbl.add_row(name, status)
            if msg_ and not ok:
                tbl.add_row("", f"[yellow]{msg_.strip()}[/yellow]")
        console.print(tbl)
    else:
        for name, ok, msg_ in results:
            status = "✓" if ok else "✗"
            print(f"{status} {name}")
            if msg_ and not ok:
                print(f"  -> {msg_.strip()}")


# --------------------------------------------------------------------------- #
# MAIN FLOW                                                                 #
# --------------------------------------------------------------------------- #
def main():
    steps: list[Tuple[str, bool, Optional[str]]] = []

    # 1. Detect torch & CUDA
    tver, cver = _torch_and_cuda()
    if not tver or not cver:
        msg("❌ Torch is not installed. Please install PyTorch first.")
        sys.exit(1)
    steps.append(("Detect torch/cuda", True, None))

    # 2. Check SageAttention version
    sage_ver = _installed_version("sageattention")
    if sage_ver:
        msg(f"SageAttention already installed: {sage_ver}")
    else:
        msg("SageAttention not found.")
    steps.append(
        ("Check SageAttention", True, None)
    )

    # 3. Decide which wheel to install
    wheel_name = _select_wheel(tver, cver)

    if wheel_name is None:
        # Upgrade required – use latest wheel
        msg("Your torch/cuda combo is older than supported.")
        msg(f"Suggested upgrade: torch = 2.9.0+cu130, cuda = 13.0")
        ans = input("Proceed with the upgrade? (y/N): ").strip().lower()
        if ans not in ("y", "yes"):
            msg("❌ Upgrade aborted by user.")
            sys.exit(1)
        wheel_name = LATEST_WHEEL
    elif (
        tver.startswith("2.9.0") and cver.startswith("13")
    ):  # already latest – no reinstall needed
        msg("Your torch/cuda combo is the newest supported one.")
        steps.append(("SageAttention up‑to‑date", True, None))
        _summary(steps)
        sys.exit(0)

    # 4. Download wheel
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            wheel_path = _download_wheel(wheel_name, Path(tmpdir))
            steps.append(("Download wheel", True, None))
        except Exception as e:
            steps.append(("Download wheel", False, str(e)))
            _summary(steps)
            sys.exit(1)

        # 5. Install wheel
        ok, out = _pip_install(wheel_path)
        if not ok:
            steps.append(("Install wheel", False, out))
            _summary(steps)
            sys.exit(1)
        steps.append(("Install wheel", True, None))

    # 6. Done – summary
    _summary(steps)
    msg("\n✅ All done!")


if __name__ == "__main__":
    main()
