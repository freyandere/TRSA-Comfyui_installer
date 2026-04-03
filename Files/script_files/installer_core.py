#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Installer - Multi-Package Accelerator
Version: 2.7.0
Author: freyandere
Repository: https://github.com/freyandere/TRSA-Comfyui_installer

CHANGELOG 2.7.0:
- Added GPU detection with nvidia-smi and torch.cuda fallback
- Replaced hardcoded wheel configs with manifest-driven resolution
- Added state backup and --restore recovery
- Remote manifest fetch from wildminder/AI-windows-whl with local fallback
- Multi-package installation (SageAttention, Triton, more)

CHANGELOG 2.8.0:
- Ported to single executable (PyInstaller --onefile)
- Rich TUI with panels, progress bars, and status colors
- CLI: --auto, --yes, --dry-run, --version, --restore flags
- Fixed pip subprocess path for frozen bundles
"""

import sys
import os
import subprocess
import re
import logging
import json
import argparse
import urllib.request
import urllib.error
import urllib.parse
import shutil
import signal
import atexit
from typing import Optional, Tuple, Dict, List, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Rich TUI imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Packaging is bundled in the exe, so no auto-install needed
try:
    from packaging import version as pkg_version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False
    print("WARNING: 'packaging' library not found.")

try:
    from installer_core_lang import get_text, get_system_language
except ImportError:
    print("ERROR: installer_core_lang.py not found!")
    input("Press Enter to exit...")
    sys.exit(1)

# ============================================================================
# RICH CONSOLE & TUI HELPERS
# ============================================================================

# When running frozen by PyInstaller, sys.executable points to the .exe itself.
# For pip operations we need the actual ComfyUI embedded python.exe.
def _get_comfyui_python() -> str:
    """Find the ComfyUI embedded Python for pip operations."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        candidate = exe_dir / "python.exe"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _get_console() -> "Console":
    """Get or create the Rich console singleton."""
    if not hasattr(_get_console, "instance"):
        _get_console.instance = Console()
    return _get_console.instance


def panel(text: str, title: str = "", border_style: str = "blue") -> None:
    """Print a Rich panel box."""
    console = _get_console()
    if HAS_RICH:
        console.print(Panel(text, title=title, border_style=border_style))
    else:
        print(f"\n{'=' * 60}")
        if title:
            print(f"  {title}")
            print(f"{'=' * 60}")
        print(text)
        print(f"{'=' * 60}\n")


def rule(title: str = "", style: str = "blue") -> None:
    """Print a Rich rule line."""
    console = _get_console()
    if HAS_RICH:
        console.rule(title=title, style=style)
    else:
        print(f"\n{'=' * 60}")
        if title:
            print(f"  {title}")


def ask(prompt: str, default: str = "") -> str:
    """Ask user for input with optional default."""
    if HAS_RICH:
        return Prompt.ask(prompt, default=default)
    if default:
        val = input(f"{prompt} [{default}]: ")
        return val if val else default
    return input(prompt + " ")


def status(text: str, style: str = "blue") -> None:
    """Print a status message with color."""
    console = _get_console()
    if HAS_RICH:
        console.print(f"[{style}]{text}[/{style}]")
    else:
        print(text)


def status_ok(text: str) -> None:
    status(f"\u2713 {text}", "green")


def status_fail(text: str) -> None:
    status(f"\u2717 {text}", "red")


def status_warn(text: str) -> None:
    status(f"\u26a0 {text}", "yellow")


def status_info(text: str) -> None:
    status(text, "blue")

# ============================================================================
# AUTOMATIC CLEANUP ON CRASH / WINDOW CLOSE
# ============================================================================

_CLEANUP_FILES: List[Path] = []


def _emergency_cleanup() -> None:
    """Remove temp files when the process is killed or crashes."""
    for f in list(_CLEANUP_FILES):
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass


def _cleanup_temp_files(files: List[Path], logger: logging.Logger) -> None:
    """Clean up temp files and register them as cleared."""
    for f in files:
        try:
            if f.exists():
                f.unlink()
                logger.debug(f"Deleted: {f}")
        except Exception as e:
            logger.warning(f"Could not delete {f}: {e}")
    _CLEANUP_FILES.clear()


atexit.register(_emergency_cleanup)

# Catch Ctrl+C and window close for cleanup
def _signal_handler(signum: int, frame) -> None:  # type: ignore[reportAny]
    _emergency_cleanup()
    sys.exit(130)


try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except (ValueError, OSError):
    # Can fail inside threads or on some platforms; not critical
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "2.7.0"
GITHUB_REPO = "https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
WHEELS_BASE_PATH = "wheels"
MIN_PYTHON_VERSION = (3, 9)

# Disk space requirements (MB)
DISK_SPACE_MIN = 100
DISK_SPACE_TORCH_UPGRADE = 3000  # PyTorch upgrade needs ~2.5GB

# Triton builds for Windows
TRITON_BASE_URL = "https://github.com/woct0rdho/triton-windows/releases/download"
TRITON_VERSIONS = {
    "py39": "v3.0.0/triton-3.0.0-cp39-cp39-win_amd64.whl",
    "py310": "v3.0.0/triton-3.0.0-cp310-cp310-win_amd64.whl",
    "py311": "v3.1.0/triton-3.1.0-cp311-cp311-win_amd64.whl",
    "py312": "v3.1.0/triton-3.1.0-cp312-cp312-win_amd64.whl",
    "py313": "v3.1.0/triton-3.1.0-cp313-cp313-win_amd64.whl",
}

# SageAttention configurations
SUPPORTED_CONFIGS: Dict[str, Dict[str, Dict[str, str]]] = {
    "py39": {
        "cu124_torch251": {
            "torch_version": "2.5.1",
            "cuda_version": "12.4",
            "wheel": "sageattention-2.2.0+cu124torch2.5.1.post2-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",  # noqa: E501
        },
        "cu126_torch260": {
            "torch_version": "2.6.0",
            "cuda_version": "12.6",
            "wheel": "sageattention-2.2.0+cu126torch2.6.0.post2-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.6.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126",  # noqa: E501
        },
        "cu128_torch271": {
            "torch_version": "2.7.1",
            "cuda_version": "12.8",
            "wheel": "sageattention-2.2.0+cu128torch2.7.1.post3-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",  # noqa: E501
        },
        "cu128_torch280": {
            "torch_version": "2.8.0",
            "cuda_version": "12.8",
            "wheel": "sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.8.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",  # noqa: E501
            "boost": "+20% FP16Fast",
        },
        "cu130_torch290": {
            "torch_version": "2.9.0",
            "cuda_version": "13.0",
            "wheel": "sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.9.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",  # noqa: E501
            "boost": "+25% speed",
        },
    },
    "py313": {
        "cu130_torch290": {
            "torch_version": "2.9.0",
            "cuda_version": "13.0",
            "wheel": "sageattention-2.2.0.post3%2Bcu130torch2.9.0-cp313-cp313-win_amd64.whl",
            "python_folder": "3.13",
            "torch_install_cmd": "torch==2.9.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",  # noqa: E501
            "boost": "+25% speed",
        },
        "cu130_torch2100": {
            "torch_version": "2.10.0",
            "cuda_version": "13.0",
            "wheel": "sageattention-2.2.0.post3%2Bcu130torch2.10.0-cp313-cp313-win_amd64.whl",
            "python_folder": "3.13",
            "torch_install_cmd": "torch==2.10.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",  # noqa: E501
            "boost": "+30% latest",
        },
    },
}

# Compute capability -> minimum CUDA version
COMPUTE_CAP_CUDA_MAP: Dict[Tuple[int, int], str] = {
    (3, 0): "11.8",
    (3, 5): "11.8",
    (3, 7): "11.8",
    (5, 0): "11.8",
    (5, 2): "11.8",
    (6, 0): "11.8",
    (6, 1): "11.8",
    (6, 2): "11.8",
    (7, 0): "11.8",
    (7, 2): "11.8",
    (7, 5): "11.8",
    (8, 0): "11.8",
    (8, 6): "11.8",
    (8, 9): "12.4",
    (9, 0): "12.0",
    (10, 0): "12.8",
    (12, 0): "13.0",
}


def _get_min_cuda_for_compute_cap(major: int, minor: int) -> Optional[str]:
    """Return the minimum CUDA version for a given compute capability."""
    for (cap_major, cap_minor), cuda_ver in sorted(
        COMPUTE_CAP_CUDA_MAP.items(), reverse=True
    ):
        if major >= cap_major:
            if major == cap_major and minor >= cap_minor:
                return cuda_ver
            if major > cap_major:
                return cuda_ver
    return None


# ============================================================================
# REMOTE MANIFEST CONFIGURATION
# ============================================================================

# Remote manifest for wheel resolution
REMOTE_MANIFEST_URL = "https://raw.githubusercontent.com/wildminder/AI-windows-whl/main/wheels.json"
MANIFEST_TIMEOUT = 5  # seconds
LOCAL_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "manifest", "fallback_wheels.json",
)

# Packages to install: (name, description, is_critical)
PACKAGES_TO_INSTALL: List[Tuple[str, str, bool]] = [
    ("sageattention", "SageAttention attention optimization", True),
    ("triton", "Triton GPU compiler", False),
]


class WheelManifest:
    """Fetches, validates, and resolves wheel manifests."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.data: Dict[str, Any] = {}
        self.source = ""  # "remote" or "local"

    def fetch(self) -> bool:
        """Try remote manifest first, fall back to local. Returns True if loaded."""
        if self._fetch_remote():
            self.source = "remote"
            return True
        if self._load_local():
            self.source = "local"
            return True
        self.logger.error("No manifest available (remote unreachable, local missing)")
        return False

    def _fetch_remote(self) -> bool:
        """Fetch and validate remote manifest."""
        try:
            self.logger.debug(f"Fetching manifest from {REMOTE_MANIFEST_URL}")
            req = urllib.request.Request(REMOTE_MANIFEST_URL)
            req.add_header("User-Agent", "TRSA-Installer")
            with urllib.request.urlopen(req, timeout=MANIFEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            if self._validate_schema(data):
                self.data = data
                self.logger.info("Remote manifest loaded and validated")
                return True
            self.logger.warning("Remote manifest schema validation failed")
        except Exception as e:
            self.logger.debug(f"Remote manifest fetch failed: {e}")
        return False

    def _load_local(self) -> bool:
        """Load local fallback manifest."""
        try:
            path = Path(LOCAL_MANIFEST_PATH)
            if not path.exists():
                self.logger.debug(f"Local manifest not found: {path}")
                return False
            data = json.loads(path.read_text(encoding="utf-8"))
            if self._validate_schema(data):
                self.data = data
                self.logger.info("Local fallback manifest loaded")
                return True
        except Exception as e:
            self.logger.debug(f"Local manifest load failed: {e}")
        return False

    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate manifest has the expected structure."""
        if not isinstance(data, dict):
            return False
        if "packages" not in data:
            return False
        if not isinstance(data["packages"], dict):
            return False
        for pkg_name, pkg_data in data["packages"].items():
            if not isinstance(pkg_data, dict):
                return False
            if "wheels" not in pkg_data:
                return False
            if not isinstance(pkg_data["wheels"], list):
                return False
            for wheel in pkg_data["wheels"]:
                if not isinstance(wheel, dict):
                    return False
                if "filename" not in wheel:
                    return False
                if "python_tags" not in wheel:
                    return False
                if "cuda_tag" not in wheel:
                    return False
                if "url_pattern" not in wheel:
                    return False
        return True

    def resolve(self, python_minor: int, cuda_ver: Optional[str],
                torch_ver: Optional[str], package_name: str) -> Optional[Dict[str, Any]]:
        """Resolve the best matching wheel for a package.

        Returns wheel dict with resolved url, or None.
        """
        pkg_data = self.data.get("packages", {}).get(package_name)
        if not pkg_data:
            return None

        cp_tag = f"cp3{python_minor}"

        candidates = []
        for wheel in pkg_data["wheels"]:
            if cp_tag not in wheel["python_tags"]:
                continue
            if wheel["cuda_tag"] != "any" and wheel["cuda_tag"] != cuda_ver:
                continue
            if wheel.get("torch_min"):
                if not torch_ver or compare_versions(torch_ver, wheel["torch_min"]) < 0:
                    continue
            candidates.append(wheel)

        if not candidates:
            return None

        candidates.sort(
            key=lambda w: parse_version_safe(w.get("torch_min", "0.0.0")),
            reverse=True,
        )
        best = candidates[0].copy()
        url_pattern = best.get("url_pattern", "")
        filename = urllib.parse.quote(best["filename"], safe="")
        if url_pattern.startswith("local://"):
            best["url"] = url_pattern
            best["is_local"] = True
        else:
            best["url"] = url_pattern.format(filename=filename)
            best["is_local"] = False
        return best


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SystemInfo:
    python_version: str
    python_tuple: Tuple[int, int, int]
    torch_version: Optional[str]
    cuda_version: Optional[str]
    sage_version: Optional[str]
    is_compatible: bool
    upgrade_needed: bool
    python_config_key: str
    # GPU fields (may be None if no GPU detected)
    gpu_name: Optional[str] = None
    vram_mb: Optional[int] = None
    compute_cap: Optional[Tuple[int, int]] = None
    driver_version: Optional[str] = None
    min_cuda: Optional[str] = None


@dataclass
class InstallationResult:
    success: bool
    previous_version: Optional[str]
    installed_version: Optional[str]
    errors: List[str]
    log_path: str


# ============================================================================
# STATE BACKUP & RESTORE
# ============================================================================

TRACKED_PACKAGES = ["torch", "sageattention", "triton", "xformers"]
BACKUP_FILE = "TRSA_state_backup.json"


def _get_package_version(package: str) -> Optional[str]:
    """Get installed version of a package via pip show, or None if not installed."""
    try:
        result = subprocess.run(
            [_get_comfyui_python(), "-m", "pip", "show", package],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def snapshot_state(logger: logging.Logger) -> Dict[str, Any]:
    """Snapshot current versions of tracked packages."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
    }
    for pkg in TRACKED_PACKAGES:
        snapshot[f"{pkg}_version"] = _get_package_version(pkg)
    logger.debug(f"State snapshot: {snapshot}")
    return snapshot


def save_backup(backup_data: Dict[str, Any], logger: logging.Logger) -> str:
    """Write state snapshot to backup file. Returns the file path."""
    try:
        backup_path = Path(BACKUP_FILE)
        backup_path.write_text(json.dumps(backup_data, indent=2), encoding="utf-8")
        logger.info(f"State backup saved to {backup_path}")
        return str(backup_path.absolute())
    except Exception as e:
        logger.warning(f"Could not write state backup: {e}")
        return ""


def restore_mode(logger: logging.Logger) -> None:
    """Restore system to backed-up state. Called when --restore flag is passed."""
    try:
        from installer_core_lang import get_text
    except ImportError:
        print("[ERROR] Cannot load translations, continuing in English only.")
        def get_text(lang, key, **kwargs):
            return key

    python = _get_comfyui_python()
    t = lambda key, **kwargs: get_text("en", key, **kwargs)

    backup_path = Path(BACKUP_FILE)
    if not backup_path.exists():
        print("[ ERROR ]")
        print(f"   No backup file found at {backup_path}.")
        print("   Cannot restore. You may need to manually uninstall packages.")
        input(t("press_enter"))
        return

    try:
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Corrupted backup: {e}")
        print("[ ERROR ]")
        print("   Backup file is corrupted or unreadable.")
        print("   Attempting best-effort cleanup...")
        for pkg in TRACKED_PACKAGES:
            print(f"   Uninstalling {pkg}...")
            subprocess.run(
                [python, "-m", "pip", "uninstall", "-y", pkg],
                capture_output=True, timeout=30,
            )
        print(f"   Done. Checked: {', '.join(TRACKED_PACKAGES)}")
        input(t("press_enter"))
        return

    print("[ Restore ]")
    print(f"   Restoring state from {backup.get('timestamp', 'unknown')}...")
    print()

    for pkg in TRACKED_PACKAGES:
        version_key = f"{pkg}_version"
        saved_version = backup.get(version_key)

        if saved_version:
            print(f"   Restoring {pkg} to {saved_version}...")
            try:
                subprocess.run(
                    [python, "-m", "pip", "uninstall", "-y", pkg],
                    capture_output=True, timeout=30,
                )
                cmd = [python, "-m", "pip", "install", f"{pkg}=={saved_version}"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    print(f"   {pkg} restored")
                else:
                    print(f"   {pkg} restore failed")
            except Exception as e:
                print(f"   {pkg} restore error: {e}")
        elif saved_version is None:
            pkg_ver = _get_package_version(pkg)
            if pkg_ver:
                print(f"   Removing {pkg} {pkg_ver} (was not installed before)...")
                try:
                    subprocess.run(
                        [python, "-m", "pip", "uninstall", "-y", pkg],
                        capture_output=True, timeout=30,
                    )
                    print(f"   {pkg} removed")
                except Exception as e:
                    print(f"   {pkg} removal error: {e}")

    print()
    print("   Restore complete.")
    print("   You can now run the installer normally to start fresh.")
    input(t("press_enter"))


# ============================================================================
# LOGGING
# ============================================================================


def setup_logging() -> Tuple[logging.Logger, str]:
    timestamp = datetime.now().strftime("%H.%M-%d.%m.%Y")
    log_filename = f"TRSA_install_{timestamp}.log"

    logger = logging.getLogger("TRSAInstaller")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_filename, encoding="utf-8", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger, str(Path(log_filename).absolute())


# ============================================================================
# VERSION COMPARISON HELPERS
# ============================================================================


def parse_version_safe(ver_str: str) -> Tuple[int, int, int]:
    """
    Parse version into a tuple (major, minor, patch).
    """
    if HAS_PACKAGING:
        try:
            parsed = pkg_version.parse(ver_str)  # noqa: F821
            if hasattr(parsed, "release") and parsed.release:
                release = parsed.release
                if len(release) >= 3:
                    return release[0], release[1], release[2]
                if len(release) == 2:
                    return release[0], release[1], 0
                if len(release) == 1:
                    return release[0], 0, 0
        except Exception:
            pass

    try:
        parts = ver_str.split(".")
        while len(parts) < 3:
            parts.append("0")
        major, minor, patch = (int(x) for x in parts[:3])
        return major, minor, patch
    except Exception:
        return (0, 0, 0)


def compare_versions(ver1: str, ver2: str) -> int:
    """
    Compare two version strings.
    Returns -1 if ver1 < ver2, 0 if equal, 1 if ver1 > ver2
    """
    v1 = parse_version_safe(ver1)
    v2 = parse_version_safe(ver2)

    if v1 < v2:
        return -1
    if v1 > v2:
        return 1
    return 0


# ============================================================================
# INSTALLER CLASS
# ============================================================================


class TRSAInstaller:
    def __init__(self) -> None:
        self.logger, self.log_path = setup_logging()
        self.lang = "en"
        self.system_info: Optional[SystemInfo] = None
        self.selected_config: Optional[Dict[str, str]] = None
        self.errors: List[str] = []
        self.temp_files: List[Path] = []
        self.installation_results: List[Dict[str, str]] = []
        self.last_backup_path = ""
        self.auto_mode = False
        self.yes_mode = False
        self.dry_run = False
        self._success = False

        self.logger.info(f"TRSA Installer v{VERSION} initialized")
        self.logger.debug(f"Log: {self.log_path}")
        self.logger.debug(f"Python: {sys.version}")
        has_packaging = HAS_PACKAGING if 'HAS_PACKAGING' in dir() else False
        if not has_packaging:
            self.logger.debug("Packaging library: NOT AVAILABLE")
        else:
            self.logger.debug("Packaging library: Available")

    def t(self, key: str, **kwargs: str) -> str:
        return get_text(self.lang, key, **kwargs)

    # ========================================================================
    # WELCOME & LANGUAGE
    # ========================================================================

    def show_welcome_screen(self) -> None:
        console = _get_console()
        title = self.t("welcome_title")
        ver = self.t("welcome_version", version=VERSION)

        content = f"  {title}\n  {ver}\n  One-click GPU acceleration"
        if HAS_RICH:
            console.print(Panel(content, title="TRSA ComfyUI Installer", border_style="cyan"))
        else:
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  {title}")
            print(f"  {ver}")
            print("  One-click GPU acceleration")
            print(f"{sep}\n")

        self.select_language()

    def select_language(self) -> None:
        if not self.auto_mode:
            panel(
                f"{self.t('lang_option_en')}\n"
                f"{self.t('lang_option_ru')}\n"
                f"{self.t('lang_default')}",
                title=self.t("lang_select_prompt"),
            )

        if self.auto_mode:
            self.lang = get_system_language()
        else:
            choice = ask(self.t("lang_choice_prompt"), default="")

            if choice == "1":
                self.lang = "en"
            elif choice == "2":
                self.lang = "ru"
            else:
                self.lang = get_system_language()
                if choice and choice not in ["1", "2", ""]:
                    status_warn(self.t("lang_invalid"))

        status_info(self.t("lang_selected"))
        self.logger.info(f"Language: {self.lang}")

    # ========================================================================
    # SYSTEM CHECKS
    # ========================================================================

    def check_system(self) -> SystemInfo:
        rule("System Compatibility Check", "blue")
        self.logger.info("System check started")

        # GPU detection (before system checks)
        gpu_info = self._detect_gpu_info()
        self._print_gpu_detection(gpu_info)

        # Python version
        py_tuple = sys.version_info[:3]
        py_ver = f"{py_tuple[0]}.{py_tuple[1]}.{py_tuple[2]}"

        if py_tuple < MIN_PYTHON_VERSION:
            status_fail(self.t("error_python_version", version=py_ver))
            self.logger.error(f"Python {py_ver} too old")
            input(self.t("press_enter"))
            sys.exit(1)

        # Minor version: 13 → py313
        py_config_key = "py313" if py_tuple[1] == 13 else "py39"

        # PyTorch & CUDA
        torch_ver, cuda_ver = self._get_torch_info()
        if not torch_ver:
            status_fail(self.t("error_torch_not_installed"))
            self.logger.error("PyTorch not found")
            input(self.t("press_enter"))
            sys.exit(1)

        # SageAttention
        sage_ver = self._get_sage_version()

        # Compatibility
        compatible, upgrade = self._check_compatibility(torch_ver, cuda_ver)

        info = SystemInfo(
            python_version=py_ver,
            python_tuple=py_tuple,
            torch_version=torch_ver,
            cuda_version=cuda_ver,
            sage_version=sage_ver,
            is_compatible=compatible,
            upgrade_needed=upgrade,
            python_config_key=py_config_key,
            gpu_name=gpu_info.get("gpu_name"),
            vram_mb=gpu_info.get("vram_mb"),
            compute_cap=gpu_info.get("compute_cap"),
            driver_version=gpu_info.get("driver_version"),
            min_cuda=gpu_info.get("min_cuda"),
        )
        self.system_info = info

        # Build Rich card
        sage_display = sage_ver if sage_ver else "Not installed"
        status_text = (
            "Fully compatible" if compatible and not upgrade
            else "Upgrade recommended" if upgrade
            else "Needs attention"
        )

        lines = [
            f"GPU:        {info.gpu_name or 'Not detected'}"
            + (f" ({info.vram_mb / 1024:.1f} GB)" if info.vram_mb else ""),
            f"Python:     {py_ver}",
            f"PyTorch:    {torch_ver}",
            f"CUDA:       {cuda_ver or 'N/A'}",
            f"Sage:       {sage_display}",
            "",
            f"Status:     {status_text}",
        ]
        text = "\n".join(lines)

        style = "green" if (compatible and not upgrade) else "yellow"
        panel(text, title="System Compatibility", border_style=style)

        self.logger.info("System check complete")
        return info

    def _get_torch_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get PyTorch & CUDA versions as strings."""
        try:
            import torch  # type: ignore

            torch_full = torch.__version__
            torch_ver = torch_full.split("+")[0]

            try:
                nvcc = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if nvcc.returncode == 0:
                    match = re.search(r"release (\d+\.\d+)", nvcc.stdout)
                    if match:
                        cuda_ver = match.group(1)
                        self.logger.debug(f"CUDA from nvcc: {cuda_ver}")
                        return torch_ver, cuda_ver
            except Exception:
                pass

            match = re.match(r"(\d+\.\d+\.\d+)\+cu(\d+)", torch_full)
            if match:
                cuda_raw = match.group(2)
                if len(cuda_raw) == 3:
                    cuda_ver = f"{cuda_raw[:-1]}.{cuda_raw[-1]}"
                else:
                    cuda_ver = cuda_raw
                self.logger.debug(f"CUDA from torch: {cuda_ver}")
                return torch_ver, cuda_ver

            return torch_ver, None
        except Exception as e:
            self.logger.error(f"Failed to get torch info: {e}")
            return None, None

    def _get_sage_version(self) -> Optional[str]:
        """Get SageAttention base version."""
        try:
            result = subprocess.run(
                [self._get_comfyui_python(), "-m", "pip", "show", "sageattention"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
                        return version.split("+")[0]
        except Exception:
            pass
        return None

    def _check_compatibility(
        self, torch_ver: str, cuda_ver: Optional[str]
    ) -> Tuple[bool, bool]:
        """
        Exact match detection + proper upgrade logic.
        Returns: (is_compatible, needs_upgrade)
        """
        if not torch_ver:
            return False, False

        torch_cmp_290 = compare_versions(torch_ver, "2.9.0")
        torch_cmp_251 = compare_versions(torch_ver, "2.5.1")

        if torch_cmp_290 == 0 and cuda_ver == "13.0":
            self.logger.debug("Perfect match: Torch 2.9.0 + CUDA 13.0 (no upgrade)")
            return True, False

        if torch_cmp_290 > 0 and cuda_ver == "13.0":
            self.logger.debug(f"Newer: Torch {torch_ver} > 2.9.0 (no upgrade)")
            return True, False

        if torch_cmp_251 >= 0:
            needs_upgrade = torch_cmp_290 < 0
            self.logger.debug(
                f"Compatible: Torch >= 2.5.1, upgrade_needed={needs_upgrade}"
            )
            return True, needs_upgrade

        self.logger.debug("Incompatible: Torch < 2.5.1 (upgrade required)")
        return False, True

    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU info via nvidia-smi with torch.cuda fallback."""
        result: Dict[str, Any] = {
            "gpu_name": None,
            "vram_mb": None,
            "compute_cap": None,
            "driver_version": None,
            "min_cuda": None,
        }

        try:
            nvsmi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,compute_cap,name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if nvsmi.returncode == 0 and nvsmi.stdout.strip():
                line = nvsmi.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    result["driver_version"] = parts[0]
                    cap_parts = parts[1].split(".")
                    if len(cap_parts) == 2:
                        cap = (int(cap_parts[0]), int(cap_parts[1]))
                        result["compute_cap"] = cap
                        result["min_cuda"] = _get_min_cuda_for_compute_cap(*cap)
                    result["gpu_name"] = parts[2]
                    vram_str = parts[3].replace(" MiB", "").strip()
                    try:
                        result["vram_mb"] = int(float(vram_str))
                    except ValueError:
                        pass
                    self.logger.debug(
                        f"GPU from nvidia-smi: {result['gpu_name']}, "
                        f"CC={result['compute_cap']}, {result['vram_mb']}MB VRAM"
                    )
                    return result
        except Exception as e:
            self.logger.debug(f"nvidia-smi failed: {e}")

        try:
            import torch  # type: ignore
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                result["compute_cap"] = torch.cuda.get_device_capability(0)
                result["min_cuda"] = _get_min_cuda_for_compute_cap(
                    *result["compute_cap"]
                )
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["vram_mb"] = int(
                    torch.cuda.get_device_properties(0).total_memory
                    / (1024 * 1024)
                )
                self.logger.debug(f"GPU from torch.cuda: {result['gpu_name']}")
                return result
        except Exception:
            pass

        self.logger.info("No NVIDIA GPU detected - running without acceleration")
        return result

    def _print_gpu_detection(self, gpu: Dict[str, Any]) -> None:
        """Print GPU detection results in user-friendly format."""
        console = _get_console()
        lines = []

        if gpu["gpu_name"]:
            vram_gb = gpu["vram_mb"] / 1024 if gpu["vram_mb"] else 0
            lines.append(f"GPU: {gpu['gpu_name']} ({vram_gb:.1f} GB)")
            if gpu["compute_cap"]:
                lines.append(
                    f"Compute capability: {gpu['compute_cap'][0]}.{gpu['compute_cap'][1]}"
                )
            if gpu["min_cuda"]:
                lines.append(f"Recommended: CUDA {gpu['min_cuda']}")
        else:
            lines.append("No NVIDIA GPU detected — running in CPU mode")
            lines.append("")
            lines.append("ComfyUI will run on CPU. Slower but fully functional.")

        text = "\n".join(lines)
        if HAS_RICH:
            console.print(Panel(text, title="GPU Detection", border_style="magenta"))
        else:
            print(f"\n[ GPU Detection ]")
            print(text)
        print()

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def check_disk_space(self, required_mb: int = DISK_SPACE_MIN) -> bool:
        """Check disk space (non-blocking)."""
        try:
            free_mb = shutil.disk_usage(Path.cwd()).free / (1024 * 1024)
            self.logger.info(
                f"Free space: {free_mb:.0f} MB (required: {required_mb} MB)"
            )

            if free_mb < required_mb:
                print(
                    self.t(
                        "error_disk_space",
                        free=f"{free_mb:.0f}",
                        required=required_mb,
                    )
                )
                self.errors.append(
                    f"Insufficient disk space: {free_mb:.0f}MB < {required_mb}MB"
                )
                return False
            return True
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            return True

    def uninstall_package(self, package: str) -> bool:
        """Uninstall package if exists."""
        if self.dry_run:
            status_info(f"[DRY RUN] Would uninstall {package}")
            return True
        try:
            check = subprocess.run(
                [self._get_comfyui_python(), "-m", "pip", "show", package],
                capture_output=True,
                timeout=10,
            )
            if check.returncode == 0:
                print(self.t("cleanup_removing_package", package=package))
                self.logger.info(f"Uninstalling {package}")
                result = subprocess.run(
                    [self._get_comfyui_python(), "-m", "pip", "uninstall", "-y", package],
                    capture_output=True,
                    timeout=30,
                )
                return result.returncode == 0
        except Exception:
            pass
        return False

    # ========================================================================
    # PYTORCH UPGRADE
    # ========================================================================

    def prompt_torch_upgrade(self) -> bool:
        if not self.system_info or not self.system_info.upgrade_needed:
            return False
        if self.auto_mode:
            return True

        rule("PyTorch Upgrade Available", "blue")

        latest = self._get_latest_config()
        if latest:
            current_torch = str(self.system_info.torch_version)
            current_cuda = (
                str(self.system_info.cuda_version)
                if self.system_info.cuda_version
                else "N/A"
            )

            boost = latest.get("boost", "Better performance")

            if HAS_RICH:
                _get_console().print(f"Current: [yellow]{current_torch}[/yellow] + CUDA {current_cuda}")
                _get_console().print(f"Recommended: [cyan]{latest['torch_version']}[/cyan] + CUDA {latest['cuda_version']}")
                _get_console().print(f"Performance: {boost}\n")
            else:
                print(f"Current: {current_torch} + CUDA {current_cuda}")
                print(f"Recommended: {latest['torch_version']} + CUDA {latest['cuda_version']}")
                print(f"Performance: {boost}\n")

        choice = ask(self.t("torch_upgrade_prompt"), default="y")
        return choice.strip().lower() in ["y", "yes", "д", "да", ""]

    def _get_latest_config(self) -> Optional[Dict[str, str]]:
        if not self.system_info:
            return None

        key = self.system_info.python_config_key
        configs = SUPPORTED_CONFIGS.get(key, {})

        if key == "py313" and "cu130_torch2100" in configs:
            return configs["cu130_torch2100"]

        return configs.get("cu130_torch290")

    def upgrade_torch(self, config: Dict[str, str]) -> bool:
        """Upgrade PyTorch with disk space check."""
        if not self.check_disk_space(required_mb=DISK_SPACE_TORCH_UPGRADE):
            print(self.t("error_torch_disk_space"))
            self.errors.append("Insufficient disk space for PyTorch upgrade")
            return False

        if self.dry_run:
            status_info(f"[DRY RUN] Would upgrade PyTorch to {config['torch_version']}")
            return True

        print(self.t("install_torch_upgrading", version=config["torch_version"]))
        self.logger.info(
            f"Upgrading PyTorch to {config['torch_version']} (requires ~2.5GB download)"
        )

        try:
            cmd = [
                self._get_comfyui_python(),
                "-m",
                "pip",
                "install",
                "--upgrade",
                *config["torch_install_cmd"].split(),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                self.logger.info("PyTorch upgrade successful")
                if self.system_info:
                    self.system_info.torch_version = config["torch_version"]
                    self.system_info.cuda_version = config["cuda_version"]
                return True
            self.logger.error(f"Upgrade failed: {result.stderr[:200]}")
            self.errors.append("PyTorch upgrade failed")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Upgrade timeout (>10 min)")
            self.errors.append("PyTorch upgrade timeout")
            return False
        except Exception as e:
            self.logger.error(f"Upgrade error: {e}")
            self.errors.append(f"PyTorch upgrade error: {str(e)[:100]}")
            return False

    # ========================================================================
    # TRITON
    # ========================================================================

    def install_triton(self) -> bool:
        """Install Triton (optional)."""
        if self.system_info is None:
            status_fail(self.t("error_system_info_not_set"))
            self.logger.error("System info not set. Run system check first.")
            return False
        py_major, py_minor, _ = self.system_info.python_tuple

        rule(self.t("triton_title"), "blue")

        if self.auto_mode:
            pass
        else:
            choice = ask(self.t("triton_prompt"), default="y")
            if choice.strip().lower() not in ["y", "yes", "д", "да", ""]:
                print(self.t("triton_skipped"))
                return False

        return self._do_install_triton()

    def _do_install_triton(self) -> bool:
        """Internal: actually perform the Triton install (called after prompts)."""
        if self.system_info is None:
            return False
        py_major, py_minor, _ = self.system_info.python_tuple
        py_key = f"py{py_major}{py_minor}"

        if py_key == "py313":
            self.logger.info("Installing Triton for Python 3.13 via pip (triton-windows)")
            if self.dry_run:
                status_info("[DRY RUN] Would install triton-windows<3.6 via pip")
                return False
            try:
                cmd = [
                    self._get_comfyui_python(),
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "triton-windows<3.6",
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    status_ok(self.t("triton_success"))
                    self.logger.info("Triton installed via pip (triton-windows)")
                    return True
                else:
                    status_fail(self.t("triton_failed"))
                    self.logger.error(f"Triton pip install failed: {result.stderr[:200]}")
                    return False
            except Exception as e:
                status_fail(self.t("triton_failed"))
                self.logger.error(f"Triton error (pip): {e}")
                return False

        if py_key not in TRITON_VERSIONS:
            self.logger.debug(f"No Triton mapping for {py_key}")
            status_info(self.t("triton_skipped"))
            return False

        try:
            url = f"{TRITON_BASE_URL}/{TRITON_VERSIONS[py_key]}"
            filename = url.split("/")[-1]

            print(self.t("install_downloading", file=filename))
            self.logger.info(f"Downloading Triton: {url}")

            if self.dry_run:
                status_info(f"[DRY RUN] Would download {filename}")
                return False

            urllib.request.urlretrieve(url, filename)
            self.temp_files.append(Path(filename))

            print(self.t("triton_installing"))
            result = subprocess.run(
                [self._get_comfyui_python(), "-m", "pip", "install", "--upgrade", "--force-reinstall", filename],
                capture_output=True,
                timeout=120,
            )

            if result.returncode == 0:
                status_ok(self.t("triton_success"))
                self.logger.info("Triton installed from wheel")
                return True

            status_fail(self.t("triton_failed"))
            self.logger.error(f"Triton failed: {result.stderr[:200]}")
            return False
        except Exception as e:
            status_fail(self.t("triton_failed"))
            import traceback
            self.logger.error(f"Triton error: {e}\n{traceback.format_exc()}")
            return False

    # ========================================================================
    # SAGEATTENTION
    # =========================================================================

    def select_wheel_config(self) -> Optional[Dict[str, str]]:
        if not self.system_info:
            return None

        torch_ver = self.system_info.torch_version
        cuda_ver = self.system_info.cuda_version
        py_key = self.system_info.python_config_key

        print(f"{self.t('install_title')}\n{'=' * 70}")
        print(
            self.t(
                "install_selecting_wheel",
                torch=torch_ver,
                cuda=cuda_ver or "N/A",
                python=self.system_info.python_version,
            )
        )

        configs = SUPPORTED_CONFIGS.get(py_key, {})

        # Exact match
        for name, cfg in configs.items():
            torch_match = (
                compare_versions(torch_ver, cfg["torch_version"]) >= 0
                and torch_ver.startswith(cfg["torch_version"][:3])
            )

            if torch_match and cuda_ver == cfg["cuda_version"]:
                print(self.t("install_wheel_found", wheel=cfg["wheel"]))
                self.logger.info(f"Exact match: {name}")
                self.selected_config = cfg
                return cfg

        # Compatible match
        compatible = self._find_compatible(configs, torch_ver, cuda_ver)
        if compatible:
            print(self.t("install_wheel_found", wheel=compatible["wheel"]))
            self.selected_config = compatible
            return compatible

        # Fallback py39 (abi3)
        if py_key == "py313":
            configs_py39 = SUPPORTED_CONFIGS.get("py39", {})
            for cfg in configs_py39.values():
                if cuda_ver == cfg["cuda_version"]:
                    print(self.t("install_wheel_found", wheel=cfg["wheel"]))
                    self.logger.info("Using py39 fallback (abi3 compatible)")
                    self.selected_config = cfg
                    return cfg

        print(self.t("install_wheel_not_found"))
        self.logger.error("No compatible wheel")
        return None

    def _find_compatible(
        self, configs: Dict[str, Dict[str, str]], torch_ver: str, cuda_ver: Optional[str]
    ) -> Optional[Dict[str, str]]:
        if not cuda_ver:
            return None

        matches = [cfg for cfg in configs.values() if cfg["cuda_version"] == cuda_ver]
        if matches:
            matches.sort(
                key=lambda x: parse_version_safe(x["torch_version"]), reverse=True
            )
            return matches[0]
        return None

    def download_wheel(self, config: Dict[str, str]) -> Optional[Path]:
        wheel = config["wheel"]
        folder = config["python_folder"]

        encoded = urllib.parse.quote(wheel)
        url = f"{GITHUB_REPO}/{WHEELS_BASE_PATH}/{folder}/{encoded}"
        local = Path(wheel)

        print(self.t("install_downloading", file=wheel))
        self.logger.info(f"URL: {url}")

        try:
            urllib.request.urlretrieve(url, local)
            if local.exists() and local.stat().st_size > 0:
                self.temp_files.append(local)
                _CLEANUP_FILES.append(local)
                self.logger.info(f"Downloaded: {local.stat().st_size} bytes")
                return local
        except urllib.error.URLError as e:
            self.logger.error(f"Download failed (network): {e}")
            print(self.t("error_download_failed", file=wheel))
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            print(self.t("error_download_failed", file=wheel))

        return None

    def install_sageattention(self, wheel_path: Path) -> bool:
        print(self.t("install_installing"))
        self.logger.info(f"Installing: {wheel_path}")

        if self.dry_run:
            status_info(f"[DRY RUN] Would install SageAttention from {wheel_path}")
            return True

        try:
            cmd = [
                self._get_comfyui_python(),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                str(wheel_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode == 0:
                print(self.t("install_success"))
                self.logger.info("Installation successful")
                return True

            print(self.t("install_failed"))
            self.logger.error(f"Failed: {result.stderr[:200]}")
            self.errors.append("Installation failed")
            return False
        except Exception as e:
            print(self.t("install_failed"))
            self.logger.error(f"Error: {e}")
            return False

    # ========================================================================
    # MANIFEST-DRIVEN PACKAGE INSTALLATION
    # ========================================================================

    def install_packages_from_manifest(self) -> List[Dict[str, str]]:
        """Install all packages from PACKAGES_TO_INSTALL using manifest."""
        if not self.system_info:
            return []

        manifest = WheelManifest(self.logger)
        loaded = manifest.fetch()
        if not loaded:
            status_warn("Could not load any wheel manifest. Using local fallback.")

        if manifest.source == "remote":
            status_info("Checking latest wheels from wildminder/AI-windows-whl...")
        else:
            status_info("Remote unavailable. Using local fallback manifest...")
        print()

        py_minor = self.system_info.python_tuple[1]
        cuda_ver = self.system_info.cuda_version
        torch_ver = self.system_info.torch_version

        results = []

        if HAS_RICH and not self.dry_run:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                for pkg_name, pkg_desc, is_critical in PACKAGES_TO_INSTALL:
                    task_desc = f"Resolving {pkg_name}"
                    task = progress.add_task(task_desc, total=100)
                    progress.update(task, advance=20)

                    wheel = manifest.resolve(py_minor, cuda_ver, torch_ver, pkg_name)
                    progress.update(task, advance=30)

                    if not wheel:
                        progress.update(task, completed=100)
                        if is_critical:
                            status_fail(f"{pkg_name}: Not available!")
                        else:
                            status_info(f"{pkg_name} — not available for this config (safe to skip)")
                        results.append({"name": pkg_name, "status": "skipped", "error": "no matching wheel"})
                        continue

                    status_ok(f"{pkg_name} found (CUDA {wheel.get('cuda_tag', 'any')})")
                    progress.update(task, advance=30)

                    installed = self._install_wheel(wheel, pkg_name)
                    progress.update(task, completed=100)
                    results.append(installed)
        else:
            # Dumb terminal fallback or dry-run
            for pkg_name, pkg_desc, is_critical in PACKAGES_TO_INSTALL:
                print(f"   Resolving {pkg_desc}...")
                wheel = manifest.resolve(py_minor, cuda_ver, torch_ver, pkg_name)
                if not wheel:
                    if is_critical:
                        status_fail(f"{pkg_name}: Not available!")
                    else:
                        status_info(f"{pkg_name} — not available for this config")
                    results.append({"name": pkg_name, "status": "skipped", "error": "no matching wheel"})
                    continue
                status_ok(f"{pkg_name} found")
                installed = self._install_wheel(wheel, pkg_name)
                results.append(installed)

        return results

    def _install_wheel(self, wheel: Dict[str, Any], pkg_name: str) -> Dict[str, str]:
        """Download and install a single wheel."""
        filename = wheel["filename"]
        is_local = wheel.get("is_local", False)

        if is_local:
            python_folder = wheel.get("python_folder", "3.9")
            wheels_base = Path(__file__).parent.parent / "wheels" / python_folder
            local_path = wheels_base / filename
            if local_path.exists():
                self.logger.info(f"Using local wheel: {local_path}")
                return self._pip_install_file(str(local_path), pkg_name)
            else:
                self.logger.warning(f"Local wheel not found: {local_path}")
                return {"name": pkg_name, "status": "failed", "error": "local file missing"}
        else:
            url = wheel["url"]
            self.logger.info(f"Downloading {filename} from {url}")
            print(f"   Downloading {filename[:50]}... ", end="", flush=True)
            try:
                clean_name = Path(filename.split("?")[0]).name
                local = Path(clean_name)
                urllib.request.urlretrieve(url, local)
                if local.exists() and local.stat().st_size > 0:
                    self.temp_files.append(local)
                    _CLEANUP_FILES.append(local)
                    print("done")
                    return self._pip_install_file(str(local), pkg_name)
                else:
                    local.unlink(missing_ok=True)
                    return {"name": pkg_name, "status": "failed", "error": "download empty"}
            except Exception as e:
                self.logger.error(f"Download failed: {e}")
                print("failed")
                return {"name": pkg_name, "status": "failed", "error": str(e)[:100]}

    def _pip_install_file(self, filepath: str, pkg_name: str) -> Dict[str, str]:
        """Install a wheel file via pip. Returns result dict."""
        self.logger.info(f"Installing {pkg_name} from {filepath}")
        print(f"   Installing {pkg_name}... ", end="", flush=True)
        try:
            result = subprocess.run(
                [self._get_comfyui_python(), "-m", "pip", "install", "--upgrade", "--force-reinstall", filepath],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode == 0:
                ver = _get_package_version(pkg_name)
                print("done")
                return {"name": pkg_name, "status": "installed", "version": ver or "unknown"}
            else:
                self.logger.error(f"{pkg_name} install failed: {result.stderr[:200]}")
                print("failed")
                return {"name": pkg_name, "status": "failed", "error": result.stderr[:200]}
        except subprocess.TimeoutExpired:
            print("timed out")
            return {"name": pkg_name, "status": "failed", "error": "timeout"}
        except Exception as e:
            self.logger.error(f"{pkg_name} pip install error: {e}")
            print("error")
            return {"name": pkg_name, "status": "failed", "error": str(e)[:100]}

    # ========================================================================
    # ROLLBACK
    # ========================================================================

    def prompt_rollback(self) -> bool:
        if not self.system_info or not self.system_info.sage_version:
            return False
        if self.auto_mode:
            return False

        print(f"{self.t('rollback_title')}\n{'=' * 70}")
        choice = input(self.t("rollback_prompt")).strip().lower()
        return choice in ["y", "yes", "д", "да", ""]

    def rollback_sageattention(self) -> bool:
        if not self.system_info or not self.system_info.sage_version:
            return False

        prev_ver = self.system_info.sage_version
        print(self.t("rollback_starting"))
        self.logger.info(f"Rollback to {prev_ver}")

        if self.dry_run:
            status_info(f"[DRY RUN] Would rollback SageAttention to {prev_ver}")
            return True

        try:
            cmd = [
                self._get_comfyui_python(),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                f"sageattention=={prev_ver}",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode == 0:
                print(self.t("rollback_success"))
                return True

            print(self.t("rollback_failed"))
            self.logger.error(f"Rollback failed: {result.stderr[:200]}")
            return False
        except Exception:
            print(self.t("rollback_failed"))
            return False

    # ========================================================================
    # CLEANUP & SUMMARY
    # ========================================================================

    def cleanup(self) -> None:
        """Clean up temp files and register them as cleared."""
        _cleanup_temp_files(self.temp_files, self.logger)
        self.show_summary(getattr(self, "_success", False))

    def show_summary(self, success: bool) -> None:
        self._success = success
        rule(self.t("summary_title"), "green" if success else "red")

        if success:
            status_ok(self.t("summary_success"))
        else:
            status_fail(self.t("summary_failed"))

        print()

        if self.system_info:
            if self.system_info.gpu_name:
                vram = f" ({self.system_info.vram_mb / 1024:.1f} GB)" if self.system_info.vram_mb else ""
                status_info(f"GPU: {self.system_info.gpu_name}{vram}")
            else:
                status_info("GPU: Not detected (CPU mode)")

            if self.system_info.sage_version:
                status_info(f"Previous Sage version: {self.system_info.sage_version}")

            new_ver = self._get_sage_version()
            if new_ver:
                status_ok(f"Installed Sage version: {new_ver}")

            status_info(f"Python: {self.system_info.python_version}")
            status_info(f"PyTorch: {self.system_info.torch_version or 'N/A'}")
            status_info(f"CUDA: {self.system_info.cuda_version or 'N/A'}")

        if self.installation_results:
            print()
            rule("Package Results", "blue")
            for r in self.installation_results:
                if r["status"] == "installed":
                    status_ok(f"{r['name']} {r.get('version', '')}")
                elif r["status"] == "skipped":
                    status_warn(f"{r['name']} — skipped (not available for your config)")
                else:
                    status_fail(f"{r['name']} — failed: {r.get('error', 'unknown')}")
            print()

        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                status_fail(f"  {i}. {err[:100]}")

        status_info(self.t("summary_log_saved", path=self.log_path))

        if success:
            print(f"\n{self.t('summary_next_steps')}")
            print(self.t("summary_next_step_1"))
            print(self.t("summary_next_step_2"))
            print(self.t("summary_next_step_3"))
            if getattr(self, "last_backup_path", ""):
                print("  To restore your previous state, run: TRSA_installer.exe --restore")

        rule("", style="blue")

    # ========================================================================
    # MAIN WORKFLOW
    # ========================================================================

    def run(self) -> InstallationResult:
        try:
            self.logger.info("=== Stage 1: Welcome ===")
            self.show_welcome_screen()

            self.logger.info("=== Stage 2: Initial Disk Check ===")
            if not self.check_disk_space():
                self.show_summary(False)
                return self._create_result(False, None, None)

            self.logger.info("=== Stage 3: System Check ===")
            info = self.check_system()
            prev_sage = info.sage_version

            self.logger.info("=== Stage 4: State Backup ===")
            state = snapshot_state(self.logger)
            self.last_backup_path = save_backup(state, self.logger)
            if self.last_backup_path:
                print(f"   State backup saved: {self.last_backup_path}")

            if info.upgrade_needed:
                self.logger.info("=== Stage 5: PyTorch Upgrade ===")
                if self.prompt_torch_upgrade():
                    latest = self._get_latest_config()
                    if latest:
                        upgrade_success = self.upgrade_torch(latest)
                        if not upgrade_success:
                            print(self.t("torch_upgrade_continue"))

            self.logger.info("=== Stage 7: Cleanup ===")
            self.uninstall_package("triton")
            self.uninstall_package("sageattention")

            self.logger.info("=== Stage 8: Manifest-driven Installation ===")
            pkg_results = self.install_packages_from_manifest()
            self.installation_results = pkg_results

            # Determine overall success - fail only if a critical package failed
            critical_names = {name for name, _, crit in PACKAGES_TO_INSTALL if crit}
            any_critical_failed = any(
                r["status"] == "failed" and r["name"] in critical_names
                for r in pkg_results
            )

            success = not any_critical_failed
            new_sage = None
            for r in pkg_results:
                if r["name"] == "sageattention":
                    new_sage = r.get("version")

            if not success:
                sage_result = next((r for r in pkg_results if r["name"] == "sageattention"), None)
                if sage_result and sage_result["status"] == "failed":
                    if self.prompt_rollback():
                        self.rollback_sageattention()

                self.cleanup()
                self.show_summary(False)
                return self._create_result(False, prev_sage, new_sage)

            self.cleanup()
            self.show_summary(True)

            return self._create_result(True, prev_sage, new_sage)

        except KeyboardInterrupt:
            self.logger.warning("Cancelled by user")
            print("\n" + self.t("summary_failed"))
            self.cleanup()
            return self._create_result(False, None, None)
        except Exception as e:
            self.logger.critical(f"Critical error: {e}", exc_info=True)
            self.errors.append(f"Critical: {str(e)}")
            self.cleanup()
            self.show_summary(False)
            return self._create_result(False, None, None)

    def _create_result(
        self,
        success: bool,
        prev: Optional[str],
        new: Optional[str],
    ) -> InstallationResult:
        return InstallationResult(
            success=success,
            previous_version=prev,
            installed_version=new,
            errors=self.errors.copy(),
            log_path=self.log_path,
        )


# ============================================================================
# ENTRY POINT
# ============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRSA ComfyUI Installer",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Run with all defaults, no prompts",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Auto-approve all yes/no prompts",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Restore previous system state",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.version:
        print(f"TRSA ComfyUI Installer v{VERSION}")
        sys.exit(0)

    if args.restore:
        logger, log_path = setup_logging()
        console = _get_console()
        if HAS_RICH:
            console.print(f"[blue]Restore log:[/blue] {log_path}")
        print(f"Restore log: {log_path}")
        print()
        restore_mode(logger)
        sys.exit(0)

    try:
        installer = TRSAInstaller()
        installer.auto_mode = args.auto or args.yes
        installer.yes_mode = args.yes
        installer.dry_run = args.dry_run
        result = installer.run()
        input(installer.t("press_enter"))
        sys.exit(0 if result.success else 1)
    except Exception as e:
        console = _get_console()
        if HAS_RICH:
            console.print(f"\n[red]CRITICAL ERROR: {e}[/red]")
        else:
            print(f"\nCRITICAL ERROR: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
