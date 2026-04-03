#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Installer - Triton + SageAttention Accelerator
Version: 2.6.1
Author: freyandere
Repository: https://github.com/freyandere/TRSA-Comfyui_installer

CHANGELOG 2.6.1:
- Fixed Python version display (tuple formatting)
- Fixed PyTorch version parsing (string not list)
- Fixed upgrade logic: exact match detection for 2.9.0+cu130
- Fixed py_config_key detection (index check)
- Fixed Triton version key generation
- Fixed _find_compatible return type
"""

import sys
import os
import subprocess
import re
import logging
import json
import urllib.request
import urllib.error
import urllib.parse
import shutil
from typing import Optional, Tuple, Dict, List, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import packaging для version comparison
try:
    from packaging import version as pkg_version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False
    print("WARNING: 'packaging' library not found. Installing...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "packaging"],
            capture_output=True,
            timeout=30,
        )
        from packaging import version as pkg_version  # type: ignore
        HAS_PACKAGING = True
    except Exception:
        print("ERROR: Could not install 'packaging'. Version checks may be inaccurate.")

try:
    from installer_core_lang import get_text, get_system_language
except ImportError:
    print("ERROR: installer_core_lang.py not found!")
    input("Press Enter to exit...")
    sys.exit(1)

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
            [sys.executable, "-m", "pip", "show", package],
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
                [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
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
                    [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
                    capture_output=True, timeout=30,
                )
                cmd = [sys.executable, "-m", "pip", "install", f"{pkg}=={saved_version}"]
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
                        [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
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

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

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
            parsed = pkg_version.parse(ver_str)
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

        self.logger.info(f"TRSA Installer v{VERSION} initialized")
        self.logger.debug(f"Log: {self.log_path}")
        self.logger.debug(f"Python: {sys.version}")
        self.logger.debug(
            f"Packaging library: {'Available' if HAS_PACKAGING else 'NOT AVAILABLE'}"
        )

    def t(self, key: str, **kwargs: str) -> str:
        return get_text(self.lang, key, **kwargs)

    # ========================================================================
    # WELCOME & LANGUAGE
    # ========================================================================

    def show_welcome_screen(self) -> None:
        sep = "=" * 70
        title = self.t("welcome_title")
        ver = self.t("welcome_version", version=VERSION)

        print(f"\n{sep}")
        print(" " * ((70 - len(title)) // 2) + title)
        print(" " * ((70 - len(ver)) // 2) + ver)
        print(f"{sep}\n")

        self.select_language()

    def select_language(self) -> None:
        print(self.t("lang_select_prompt"))
        print(f"  {self.t('lang_option_en')}\n  {self.t('lang_option_ru')}")
        print(f"  {self.t('lang_default')}\n")

        choice = input(self.t("lang_choice_prompt")).strip()

        if choice == "1":
            self.lang = "en"
        elif choice == "2":
            self.lang = "ru"
        else:
            self.lang = get_system_language()
            if choice and choice not in ["1", "2", ""]:
                print(self.t("lang_invalid"))

        print(self.t("lang_selected"))
        self.logger.info(f"Language: {self.lang}")

    # ========================================================================
    # SYSTEM CHECKS
    # ========================================================================

    def check_system(self) -> SystemInfo:
        print(f"{self.t('check_title')}\n{'=' * 70}")
        self.logger.info("System check started")

        # GPU detection (before system checks)
        gpu_info = self._detect_gpu_info()
        self._print_gpu_detection(gpu_info)

        # Python version
        py_tuple = sys.version_info[:3]
        py_ver = f"{py_tuple[0]}.{py_tuple[1]}.{py_tuple[2]}"
        print(f"Python version: {py_ver}")

        if py_tuple < MIN_PYTHON_VERSION:
            print(self.t("error_python_version", version=py_ver))
            self.logger.error(f"Python {py_ver} too old")
            input(self.t("press_enter"))
            sys.exit(1)

        # Minor version: 13 → py313
        py_config_key = "py313" if py_tuple[1] == 13 else "py39"

        # PyTorch & CUDA
        torch_ver, cuda_ver = self._get_torch_info()
        if not torch_ver:
            print(self.t("error_torch_not_installed"))
            self.logger.error("PyTorch not found")
            input(self.t("press_enter"))
            sys.exit(1)

        print(f"PyTorch version: {torch_ver}")
        if cuda_ver:
            print(f"CUDA version: {cuda_ver}")

        # SageAttention
        sage_ver = self._get_sage_version()
        if sage_ver:
            print(self.t("check_sage_installed", version=sage_ver))
        else:
            print(self.t("check_sage_not_installed"))

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

        print()
        if compatible and not upgrade:
            print(self.t("check_compatible"))
        elif upgrade:
            print(self.t("check_upgrade_needed"))
            print(
                self.t(
                    "check_current_config",
                    torch=torch_ver,
                    cuda=cuda_ver or "N/A",
                )
            )

        print("=" * 70)
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
                [sys.executable, "-m", "pip", "show", "sageattention"],
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
        print(self.t("gpu_section_title"))
        if gpu["gpu_name"]:
            print(self.t("gpu_detecting"))
            vram_gb = gpu["vram_mb"] / 1024 if gpu["vram_mb"] else 0
            print(
                self.t(
                    "gpu_found",
                    name=gpu["gpu_name"],
                    vram_gb=vram_gb,
                )
            )
            if gpu["compute_cap"]:
                print(
                    self.t(
                        "gpu_compute_cap",
                        major=gpu["compute_cap"][0],
                        minor=gpu["compute_cap"][1],
                    )
                )
            if gpu["min_cuda"]:
                print(
                    self.t(
                        "gpu_recommended_cuda",
                        cuda=gpu["min_cuda"],
                    )
                )
        else:
            print(self.t("gpu_not_found"))
            print(self.t("gpu_cpu_fallback"))
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
        try:
            check = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                timeout=10,
            )
            if check.returncode == 0:
                print(self.t("cleanup_removing_package", package=package))
                self.logger.info(f"Uninstalling {package}")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", package],
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

        print(f"{self.t('torch_upgrade_title')}\n{'=' * 70}")

        latest = self._get_latest_config()
        if latest:
            current_torch = str(self.system_info.torch_version)
            current_cuda = (
                str(self.system_info.cuda_version)
                if self.system_info.cuda_version
                else "N/A"
            )

            boost = latest.get("boost", "Better performance")
            print(
                self.t(
                    "torch_upgrade_msg",
                    current=current_torch,
                    cuda=current_cuda,
                )
            )
            print(
                self.t(
                    "torch_upgrade_recommend",
                    target=latest["torch_version"],
                    cuda_target=latest["cuda_version"],
                )
            )
            print(f"   Performance: {boost}\n")

        choice = input(self.t("torch_upgrade_prompt")).strip().lower()
        approved = choice in ["y", "yes", "д", "да", ""]

        if approved:
            print(self.t("torch_upgrade_yes"))
        else:
            print(self.t("torch_upgrade_skip"))

        return approved

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

        print(self.t("install_torch_upgrading", version=config["torch_version"]))
        self.logger.info(
            f"Upgrading PyTorch to {config['torch_version']} (requires ~2.5GB download)"
        )

        try:
            cmd = [
                sys.executable,
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
            print(self.t("error_system_info_not_set"))
            self.logger.error("System info not set. Run system check first.")
            return False
        py_major, py_minor, _ = self.system_info.python_tuple
        py_key = f"py{py_major}{py_minor}"

        print(f"{self.t('triton_title')}\n{'=' * 70}")
        choice = input(self.t("triton_prompt")).strip().lower()

        if choice not in ["y", "yes", "д", "да", ""]:
            print(self.t("triton_skipped"))
            return False

        # Отдельная логика для Python 3.13: ставим через pip triton-windows
        if py_key == "py313":
            self.logger.info("Installing Triton for Python 3.13 via pip (triton-windows)")
            try:
                cmd = [
                    sys.executable,
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
                    print(self.t("triton_success"))
                    self.logger.info("Triton installed via pip (triton-windows)")
                    return True
                else:
                    print(self.t("triton_failed"))
                    self.logger.error(f"Triton pip install failed: {result.stderr[:200]}")
                    return False
            except Exception as e:
                print(self.t("triton_failed"))
                self.logger.error(f"Triton error (pip): {e}")
                return False

        # Для остальных версий Python — старая схема через wheel с GitHub
        if py_key not in TRITON_VERSIONS:
            self.logger.debug(f"No Triton mapping for {py_key}")
            print(self.t("triton_skipped"))
            return False

        try:
            url = f"{TRITON_BASE_URL}/{TRITON_VERSIONS[py_key]}"
            filename = url.split("/")[-1]

            print(self.t("install_downloading", file=filename))
            self.logger.info(f"Downloading Triton: {url}")

            urllib.request.urlretrieve(url, filename)
            self.temp_files.append(Path(filename))

            print(self.t("triton_installing"))
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", filename],
                capture_output=True,
                timeout=120,
            )

            if result.returncode == 0:
                print(self.t("triton_success"))
                self.logger.info("Triton installed from wheel")
                return True

            print(self.t("triton_failed"))
            self.logger.error(f"Triton failed: {result.stderr[:200]}")
            return False
        except Exception as e:
            print(self.t("triton_failed"))
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

        try:
            cmd = [
                sys.executable,
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
            print("[ WARNING ]")
            print("   Could not load any wheel manifest.")
            print("   Using local fallback — some packages may not be available.")
            print()

        if manifest.source == "remote":
            print("[ Package Resolution ]")
            print("   Checking latest wheels from wildminder/AI-windows-whl...")
        else:
            print("[ Package Resolution ]")
            print("   Remote unavailable. Using local fallback manifest...")
        print()

        py_minor = self.system_info.python_tuple[1]
        cuda_ver = self.system_info.cuda_version
        torch_ver = self.system_info.torch_version

        results = []
        for pkg_name, pkg_desc, is_critical in PACKAGES_TO_INSTALL:
            print(f"   Resolving {pkg_desc}...")
            wheel = manifest.resolve(py_minor, cuda_ver, torch_ver, pkg_name)
            if not wheel:
                if is_critical:
                    print(f"   [CRITICAL] {pkg_name}: Not available!")
                else:
                    print(f"   {pkg_name} — no wheel available for your configuration.")
                    print("   This is safe to skip. ComfyUI will work normally.")
                print()
                results.append({"name": pkg_name, "status": "skipped", "error": "no matching wheel"})
                continue

            print(f"   {pkg_name} — found (CUDA {wheel.get('cuda_tag', 'any')})")
            installed = self._install_wheel(wheel, pkg_name)
            results.append(installed)
            print()

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
                [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", filepath],
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

        print(f"{self.t('rollback_title')}\n{'=' * 70}")
        choice = input(self.t("rollback_prompt")).strip().lower()
        return choice in ["y", "yes", "д", "да", ""]

    def rollback_sageattention(self) -> bool:
        if not self.system_info or not self.system_info.sage_version:
            return False

        prev_ver = self.system_info.sage_version
        print(self.t("rollback_starting"))
        self.logger.info(f"Rollback to {prev_ver}")

        try:
            cmd = [
                sys.executable,
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
        print(self.t("cleanup_title"))
        print(self.t("cleanup_removing"))

        count = 0
        for f in self.temp_files:
            try:
                if f.exists():
                    f.unlink()
                    count += 1
                    self.logger.debug(f"Deleted: {f}")
            except Exception as e:
                self.logger.warning(f"Could not delete {f}: {e}")

        print(self.t("cleanup_success"))
        self.logger.info(f"Cleaned {count} files")

    def show_summary(self, success: bool) -> None:
        print(f"{self.t('summary_title')}\n{'=' * 70}")

        if success:
            print(self.t("summary_success"))
        else:
            print(self.t("summary_failed"))

        print()

        if self.system_info:
            # GPU info
            if self.system_info.gpu_name:
                print(f"   GPU: {self.system_info.gpu_name}")
                if self.system_info.vram_mb:
                    print(f"   VRAM: {self.system_info.vram_mb / 1024:.1f} GB")
            else:
                print("   GPU: Not detected (CPU mode)")
            print()

            if self.system_info.sage_version:
                print(
                    self.t("summary_previous_version", version=self.system_info.sage_version)
                )

            new_ver = self._get_sage_version()
            if new_ver:
                print(self.t("summary_installed_version", version=new_ver))

            print()
            print(
                self.t("summary_python_version", version=self.system_info.python_version)
            )
            print(
                self.t("summary_torch_version", version=self.system_info.torch_version or "N/A")
            )
            print(
                self.t("summary_cuda_version", version=self.system_info.cuda_version or "N/A")
            )

        # Package installation results
        if hasattr(self, "installation_results") and self.installation_results:
            print()
            print("[ Package Results ]")
            for r in self.installation_results:
                if r["status"] == "installed":
                    print(f"   {r['name']} {r.get('version', '')}")
                elif r["status"] == "skipped":
                    print(f"   {r['name']} — skipped (not available for your config)")
                else:
                    print(f"   {r['name']} — failed: {r.get('error', 'unknown')}")
            print()

        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. {err[:100]}")

        print(self.t("summary_log_saved", path=self.log_path))

        if success:
            print(self.t("summary_next_steps"))
            print("  1. Restart ComfyUI")
            print("  2. SageAttention will be automatically used")
            print("  3. Check the log file if you encounter issues")
            if getattr(self, "last_backup_path", ""):
                print("  To restore your previous state, run: TRSA_installer.bat --restore")

        print("=" * 70)

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


def main() -> None:
    if "--restore" in sys.argv:
        logger, log_path = setup_logging()
        print(f"Restore log: {log_path}")
        print()
        restore_mode(logger)
        sys.exit(0)

    try:
        installer = TRSAInstaller()
        result = installer.run()
        input(installer.t("press_enter"))
        sys.exit(0 if result.success else 1)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
