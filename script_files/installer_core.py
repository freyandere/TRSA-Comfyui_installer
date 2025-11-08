#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Installer - Triton + SageAttention Accelerator
Version: 2.7.0
Author: freyandere
Repository: https://github.com/freyandere/TRSA-Comfyui_installer

Optimized installer with:
- Triton support
- nvcc CUDA detection
- Disk space checking
- Performance boost messaging
- Original wheel names support
"""

import sys, os, subprocess, re, logging, urllib.request, urllib.error, urllib.parse, shutil
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

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

# Triton builds for Windows
TRITON_BASE_URL = "https://github.com/woct0rdho/triton-windows/releases/download"
TRITON_VERSIONS = {
    "py39": "v3.0.0/triton-3.0.0-cp39-cp39-win_amd64.whl",
    "py310": "v3.0.0/triton-3.0.0-cp310-cp310-win_amd64.whl",
    "py311": "v3.1.0/triton-3.1.0-cp311-cp311-win_amd64.whl",
    "py312": "v3.1.0/triton-3.1.0-cp312-cp312-win_amd64.whl",
    "py313": "v3.1.0/triton-3.1.0-cp313-cp313-win_amd64.whl"
}

# SageAttention configurations
SUPPORTED_CONFIGS = {
    "py39": {
        "cu124_torch251": {
            "torch_version": "2.5.1", "cuda_version": "12.4",
            "wheel": "sageattention-2.2.0+cu124torch2.5.1.post3-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
        },
        "cu126_torch260": {
            "torch_version": "2.6.0", "cuda_version": "12.6",
            "wheel": "sageattention-2.2.0+cu126torch2.6.0.post3-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.6.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
        },
        "cu128_torch271": {
            "torch_version": "2.7.1", "cuda_version": "12.8",
            "wheel": "sageattention-2.2.0+cu128torch2.7.1.post3-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        },
        "cu128_torch280": {
            "torch_version": "2.8.0", "cuda_version": "12.8",
            "wheel": "sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.8.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
            "boost": "+20% FP16Fast"
        },
        "cu130_torch290": {
            "torch_version": "2.9.0", "cuda_version": "13.0",
            "wheel": "sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.9.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",
            "boost": "+25% speed"
        }
    },
    "py313": {
        "cu130_torch290": {
            "torch_version": "2.9.0", "cuda_version": "13.0",
            "wheel": "sageattention-2.2.0+cu130torch2.9.0.post3-cp313-cp313-win_amd64.whl",
            "python_folder": "3.13",
            "torch_install_cmd": "torch==2.9.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",
            "boost": "+25% speed"
        },
        "cu130_torch2100": {
            "torch_version": "2.10.0", "cuda_version": "13.0",
            "wheel": "sageattention-2.2.0+cu130torch2.10.0.post3-cp313-cp313-win_amd64.whl",
            "python_folder": "3.13",
            "torch_install_cmd": "torch==2.10.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",
            "boost": "+30% latest"
        }
    }
}

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

@dataclass
class InstallationResult:
    success: bool
    previous_version: Optional[str]
    installed_version: Optional[str]
    errors: List[str]
    log_path: str

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> Tuple[logging.Logger, str]:
    timestamp = datetime.now().strftime("%H.%M-%d.%m.%Y")
    log_filename = f"TRSA_install_{timestamp}.log"
    
    logger = logging.getLogger('TRSAInstaller')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)
    
    # File
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    except:
        pass
    
    return logger, str(Path(log_filename).absolute())

# ============================================================================
# INSTALLER CLASS
# ============================================================================

class TRSAInstaller:
    def __init__(self):
        self.logger, self.log_path = setup_logging()
        self.lang = 'en'
        self.system_info: Optional[SystemInfo] = None
        self.selected_config: Optional[Dict] = None
        self.errors: List[str] = []
        self.temp_files: List[Path] = []
        
        self.logger.info(f"TRSA Installer v{VERSION} initialized")
        self.logger.debug(f"Log: {self.log_path}")
        self.logger.debug(f"Python: {sys.version}")
    
    def t(self, key: str, **kwargs) -> str:
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
        
        if choice == '1':
            self.lang = 'en'
        elif choice == '2':
            self.lang = 'ru'
        else:
            self.lang = get_system_language()
            if choice and choice not in ['1', '2', '']:
                print(self.t("lang_invalid"))
        
        print(self.t("lang_selected"))
        self.logger.info(f"Language: {self.lang}")
    
    # ========================================================================
    # SYSTEM CHECKS
    # ========================================================================
    
    def check_system(self) -> SystemInfo:
        print(f"{self.t('check_title')}\n{'='*70}")
        self.logger.info("System check started")
        
        # Python version
        py_tuple = sys.version_info[:3]
        py_ver = f"{py_tuple[0]}.{py_tuple[1]}.{py_tuple[2]}"
        print(self.t("check_python", version=py_ver))
        
        if py_tuple < MIN_PYTHON_VERSION:
            print(self.t("error_python_version", version=py_ver))
            self.logger.error(f"Python {py_ver} too old")
            input(self.t("press_enter"))
            sys.exit(1)
        
        py_config_key = "py313" if py_tuple[1] == 13 else "py39"
        
        # PyTorch & CUDA
        torch_ver, cuda_ver = self._get_torch_info()
        if not torch_ver:
            print(self.t("error_torch_not_installed"))
            self.logger.error("PyTorch not found")
            input(self.t("press_enter"))
            sys.exit(1)
        
        print(self.t("check_torch", version=torch_ver))
        if cuda_ver:
            print(self.t("check_cuda", version=cuda_ver))
        
        # SageAttention
        sage_ver = self._get_sage_version()
        if sage_ver:
            print(self.t("check_sage_installed", version=sage_ver))
        else:
            print(self.t("check_sage_not_installed"))
        
        # Compatibility
        compatible, upgrade = self._check_compatibility(torch_ver, cuda_ver)
        
        info = SystemInfo(
            python_version=py_ver, python_tuple=py_tuple,
            torch_version=torch_ver, cuda_version=cuda_ver,
            sage_version=sage_ver, is_compatible=compatible,
            upgrade_needed=upgrade, python_config_key=py_config_key
        )
        self.system_info = info
        
        print()
        if compatible and not upgrade:
            print(self.t("check_compatible"))
        elif upgrade:
            print(self.t("check_upgrade_needed"))
            print(self.t("check_current_config", torch=torch_ver, cuda=cuda_ver or "N/A"))
        
        print("="*70)
        return info
    
    def _get_torch_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get PyTorch & CUDA (nvcc fallback)."""
        try:
            import torch
            torch_ver = torch.__version__.split('+')[0]
            
            # Try nvcc
            try:
                nvcc = subprocess.run(["nvcc", "--version"], 
                                     capture_output=True, text=True, timeout=5)
                if nvcc.returncode == 0:
                    match = re.search(r'release (\d+\.\d+)', nvcc.stdout)
                    if match:
                        cuda_ver = match.group(1)
                        self.logger.debug(f"CUDA from nvcc: {cuda_ver}")
                        return torch_ver, cuda_ver
            except:
                pass
            
            # Fallback: parse torch version
            match = re.match(r'(\d+\.\d+\.\d+)\+cu(\d+)', torch.__version__)
            if match:
                cuda_raw = match.group(2)
                cuda_ver = f"{cuda_raw[:-1]}.{cuda_raw[-1]}" if len(cuda_raw) == 3 else cuda_raw
                self.logger.debug(f"CUDA from torch: {cuda_ver}")
                return torch_ver, cuda_ver
            
            return torch_ver, None
        except:
            return None, None
    
    def _get_sage_version(self) -> Optional[str]:
        """Get SageAttention base version."""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "show", "sageattention"],
                                   capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        return version.split('+')[0]  # Base version only
        except:
            pass
        return None
    
    def _check_compatibility(self, torch_ver: str, cuda_ver: Optional[str]) -> Tuple[bool, bool]:
        if not torch_ver:
            return False, False
        if torch_ver >= "2.9.0" and cuda_ver == "13.0":
            return True, False
        if torch_ver >= "2.5.1":
            return True, torch_ver < "2.9.0"
        return False, True
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def check_disk_space(self, required_mb: int = 100) -> bool:
        """Check disk space (non-blocking)."""
        try:
            free_mb = shutil.disk_usage(Path.cwd()).free / (1024 * 1024)
            self.logger.info(f"Free space: {free_mb:.0f} MB")
            if free_mb < required_mb:
                print(self.t("error_disk_space", free=f"{free_mb:.0f}", required=required_mb))
                return False
            return True
        except:
            return True
    
    def uninstall_package(self, package: str) -> bool:
        """Uninstall package if exists."""
        try:
            check = subprocess.run([sys.executable, "-m", "pip", "show", package],
                                  capture_output=True, timeout=10)
            if check.returncode == 0:
                print(self.t("cleanup_removing_package", package=package))
                self.logger.info(f"Uninstalling {package}")
                result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package],
                                       capture_output=True, timeout=30)
                return result.returncode == 0
        except:
            pass
        return False
    
    # ========================================================================
    # PYTORCH UPGRADE
    # ========================================================================
    
    def prompt_torch_upgrade(self) -> bool:
        if not self.system_info or not self.system_info.upgrade_needed:
            return False
        
        print(f"{self.t('torch_upgrade_title')}\n{'='*70}")
        
        latest = self._get_latest_config()
        if latest:
            boost = latest.get("boost", "Better performance")
            print(self.t("torch_upgrade_msg", 
                         current=self.system_info.torch_version,
                         cuda=self.system_info.cuda_version or "N/A"))
            print(self.t("torch_upgrade_recommend", 
                         target=latest["torch_version"], 
                         cuda_target=latest["cuda_version"]))
            print(f"   Performance: {boost}\n")
        
        choice = input(self.t("torch_upgrade_prompt")).strip().lower()
        approved = choice in ['y', 'yes', 'д', 'да', '']
        
        if approved:
            print(self.t("torch_upgrade_yes"))
        else:
            print(self.t("torch_upgrade_skip"))
        
        return approved
    
    def _get_latest_config(self) -> Optional[Dict]:
        if not self.system_info:
            return None
        
        key = self.system_info.python_config_key
        configs = SUPPORTED_CONFIGS.get(key, {})
        
        if key == "py313" and "cu130_torch2100" in configs:
            return configs["cu130_torch2100"]
        
        return configs.get("cu130_torch290")
    
    def upgrade_torch(self, config: Dict) -> bool:
        print(self.t("install_torch_upgrading", version=config["torch_version"]))
        self.logger.info(f"Upgrading PyTorch to {config['torch_version']}")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
                   *config["torch_install_cmd"].split()]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                self.logger.info("PyTorch upgrade successful")
                if self.system_info:
                    self.system_info.torch_version = config["torch_version"]
                    self.system_info.cuda_version = config["cuda_version"]
                return True
            else:
                self.logger.error(f"Upgrade failed: {result.stderr[:200]}")
                self.errors.append(f"PyTorch upgrade failed")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Upgrade timeout")
            return False
        except Exception as e:
            self.logger.error(f"Upgrade error: {e}")
            return False
    
    # ========================================================================
    # TRITON
    # ========================================================================
    
    def install_triton(self) -> bool:
        """Install Triton (optional)."""
        py_ver = f"py{self.system_info.python_tuple[0]}{self.system_info.python_tuple[1]}"
        if py_ver not in TRITON_VERSIONS:
            self.logger.debug(f"No Triton for {py_ver}")
            return False
        
        print(f"{self.t('triton_title')}\n{'='*70}")
        choice = input(self.t("triton_prompt")).strip().lower()
        
        if choice not in ['y', 'yes', 'д', 'да', '']:
            print(self.t("triton_skipped"))
            return False
        
        try:
            url = f"{TRITON_BASE_URL}/{TRITON_VERSIONS[py_ver]}"
            filename = url.split('/')[-1]
            
            print(self.t("install_downloading", file=filename))
            self.logger.info(f"Downloading Triton: {url}")
            
            urllib.request.urlretrieve(url, filename)
            self.temp_files.append(Path(filename))
            
            print(self.t("triton_installing"))
            result = subprocess.run([sys.executable, "-m", "pip", "install", 
                                    "--force-reinstall", filename],
                                   capture_output=True, timeout=120)
            
            if result.returncode == 0:
                print(self.t("triton_success"))
                self.logger.info("Triton installed")
                return True
            else:
                print(self.t("triton_failed"))
                self.logger.error(f"Triton failed: {result.stderr[:200]}")
                return False
        except Exception as e:
            print(self.t("triton_failed"))
            self.logger.error(f"Triton error: {e}")
            return False
    
    # ========================================================================
    # SAGEATTENTION
    # ========================================================================
    
    def select_wheel_config(self) -> Optional[Dict]:
        if not self.system_info:
            return None
        
        torch_ver = self.system_info.torch_version
        cuda_ver = self.system_info.cuda_version
        py_key = self.system_info.python_config_key
        
        print(f"{self.t('install_title')}\n{'='*70}")
        print(self.t("install_selecting_wheel", 
                     torch=torch_ver, cuda=cuda_ver or "N/A",
                     python=self.system_info.python_version))
        
        configs = SUPPORTED_CONFIGS.get(py_key, {})
        
        # Exact match
        for name, cfg in configs.items():
            if torch_ver.startswith(cfg["torch_version"][:3]) and cuda_ver == cfg["cuda_version"]:
                print(self.t("install_wheel_found", wheel=cfg["wheel"]))
                self.logger.info(f"Match: {name}")
                self.selected_config = cfg
                return cfg
        
        # Compatible match
        compatible = self._find_compatible(configs, torch_ver, cuda_ver)
        if compatible:
            print(self.t("install_wheel_found", wheel=compatible["wheel"]))
            self.selected_config = compatible
            return compatible
        
        # Fallback py39
        if py_key == "py313":
            configs_py39 = SUPPORTED_CONFIGS.get("py39", {})
            for cfg in configs_py39.values():
                if cuda_ver == cfg["cuda_version"]:
                    print(self.t("install_wheel_found", wheel=cfg["wheel"]))
                    self.logger.info("Using py39 fallback")
                    self.selected_config = cfg
                    return cfg
        
        print(self.t("install_wheel_not_found"))
        self.logger.error("No compatible wheel")
        return None
    
    def _find_compatible(self, configs: Dict, torch_ver: str, cuda_ver: Optional[str]) -> Optional[Dict]:
        if not cuda_ver:
            return None
        
        matches = [cfg for cfg in configs.values() if cfg["cuda_version"] == cuda_ver]
        if matches:
            matches.sort(key=lambda x: x["torch_version"], reverse=True)
            return matches[0]
        return None
    
    def download_wheel(self, config: Dict) -> Optional[Path]:
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
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            print(self.t("error_download_failed", file=wheel))
        
        return None
    
    def install_sageattention(self, wheel_path: Path) -> bool:
        print(self.t("install_installing"))
        self.logger.info(f"Installing: {wheel_path}")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", 
                   "--upgrade", "--force-reinstall", str(wheel_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print(self.t("install_success"))
                self.logger.info("Installation successful")
                return True
            else:
                print(self.t("install_failed"))
                self.logger.error(f"Failed: {result.stderr[:200]}")
                self.errors.append("Installation failed")
                return False
        except Exception as e:
            print(self.t("install_failed"))
            self.logger.error(f"Error: {e}")
            return False
    
    # ========================================================================
    # ROLLBACK
    # ========================================================================
    
    def prompt_rollback(self) -> bool:
        if not self.system_info or not self.system_info.sage_version:
            return False
        
        print(f"{self.t('rollback_title')}\n{'='*70}")
        choice = input(self.t("rollback_prompt")).strip().lower()
        return choice in ['y', 'yes', 'д', 'да', '']
    
    def rollback_sageattention(self) -> bool:
        if not self.system_info or not self.system_info.sage_version:
            return False
        
        prev_ver = self.system_info.sage_version
        print(self.t("rollback_starting"))
        self.logger.info(f"Rollback to {prev_ver}")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", 
                   "--force-reinstall", "--no-deps", f"sageattention=={prev_ver}"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print(self.t("rollback_success"))
                return True
            else:
                print(self.t("rollback_failed"))
                self.logger.error(f"Rollback failed: {result.stderr[:200]}")
                return False
        except:
            print(self.t("rollback_failed"))
            return False
    
    # ========================================================================
    # CLEANUP & SUMMARY
    # ========================================================================
    
    def cleanup(self) -> None:
        print(f"{self.t('cleanup_title')}")
        print(self.t("cleanup_removing"))
        
        count = 0
        for f in self.temp_files:
            try:
                if f.exists():
                    f.unlink()
                    count += 1
            except:
                pass
        
        print(self.t("cleanup_success"))
        self.logger.info(f"Cleaned {count} files")
    
    def show_summary(self, success: bool) -> None:
        print(f"{self.t('summary_title')}\n{'='*70}")
        
        if success:
            print(self.t("summary_success"))
        else:
            print(self.t("summary_failed"))
        
        print()
        
        if self.system_info:
            if self.system_info.sage_version:
                print(self.t("summary_previous_version", version=self.system_info.sage_version))
            
            new_ver = self._get_sage_version()
            if new_ver:
                print(self.t("summary_installed_version", version=new_ver))
            
            print()
            print(self.t("summary_python_version", version=self.system_info.python_version))
            print(self.t("summary_torch_version", version=self.system_info.torch_version or "N/A"))
            print(self.t("summary_cuda_version", version=self.system_info.cuda_version or "N/A"))
        
        if self.errors:
            print(f"\n{self.t('summary_errors', count=len(self.errors))}")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. {err[:100]}")
        
        print(f"\n{self.t('summary_log_saved', path=self.log_path)}")
        
        if success:
            print(self.t("summary_next_steps"))
            print(f"  {self.t('summary_next_step_1')}")
            print(f"  {self.t('summary_next_step_2')}")
            print(f"  {self.t('summary_next_step_3')}")
        
        print("="*70)
    
    # ========================================================================
    # MAIN WORKFLOW
    # ========================================================================
    
    def run(self) -> InstallationResult:
        try:
            # 1. Welcome
            self.logger.info("=== Stage 1: Welcome ===")
            self.show_welcome_screen()
            
            # 2. Disk check
            self.logger.info("=== Stage 2: Disk Check ===")
            if not self.check_disk_space():
                self.show_summary(False)
                return self._create_result(False, None, None)
            
            # 3. System check
            self.logger.info("=== Stage 3: System Check ===")
            info = self.check_system()
            prev_sage = info.sage_version
            
            # 4. PyTorch upgrade
            if info.upgrade_needed:
                self.logger.info("=== Stage 4: PyTorch Upgrade ===")
                if self.prompt_torch_upgrade():
                    latest = self._get_latest_config()
                    if latest:
                        self.upgrade_torch(latest)
            
            # 5. Cleanup old packages
            self.logger.info("=== Stage 5: Cleanup ===")
            self.uninstall_package("triton")
            self.uninstall_package("sageattention")
            
            # 6. Triton
            self.logger.info("=== Stage 6: Triton ===")
            self.install_triton()
            
            # 7. Select SageAttention
            self.logger.info("=== Stage 7: Selection ===")
            config = self.select_wheel_config()
            if not config:
                self.show_summary(False)
                return self._create_result(False, prev_sage, None)
            
            # 8. Download
            self.logger.info("=== Stage 8: Download ===")
            wheel = self.download_wheel(config)
            if not wheel:
                self.show_summary(False)
                return self._create_result(False, prev_sage, None)
            
            # 9. Install
            self.logger.info("=== Stage 9: Install ===")
            success = self.install_sageattention(wheel)
            
            if not success:
                if self.prompt_rollback():
                    self.rollback_sageattention()
                self.cleanup()
                self.show_summary(False)
                return self._create_result(False, prev_sage, None)
            
            new_ver = self._get_sage_version()
            self.cleanup()
            self.show_summary(True)
            
            return self._create_result(True, prev_sage, new_ver)
            
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
    
    def _create_result(self, success: bool, prev: Optional[str], new: Optional[str]) -> InstallationResult:
        return InstallationResult(
            success=success,
            previous_version=prev,
            installed_version=new,
            errors=self.errors.copy(),
            log_path=self.log_path
        )

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
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
