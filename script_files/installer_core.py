#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI SageAttention Installer - Core Module
Version: 2.6.0
Author: freyandere
Repository: https://github.com/freyandere/TRSA-Comfyui_installer

Production-ready installer with comprehensive error handling,
logging, multi-version support, and rollback capabilities.
"""

import sys
import os
import subprocess
import re
import logging
import urllib.request
import urllib.error
import shutil
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import localization module
try:
    from installer_core_lang import get_text, get_system_language, TRANSLATIONS
except ImportError:
    print("CRITICAL ERROR: installer_core_lang.py not found!")
    print("Please ensure both installer files are downloaded correctly.")
    input("\nPress Enter to exit...")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "2.6.0"
GITHUB_REPO = "https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
WHEELS_BASE_PATH = "wheels"

# Supported configurations organized by Python version
SUPPORTED_CONFIGS = {
    # Python 3.9+ (cp39-abi3) - Universal wheels compatible with 3.9, 3.10, 3.11, 3.12
    "py39": {
        "cu124_torch251": {
            "torch_version": "2.5.1",
            "cuda_version": "12.4",
            "wheel": "sage_cu124_torch251_py39.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
        },
        "cu126_torch260": {
            "torch_version": "2.6.0",
            "cuda_version": "12.6",
            "wheel": "sage_cu126_torch260_py39.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.6.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
        },
        "cu128_torch271": {
            "torch_version": "2.7.1",
            "cuda_version": "12.8",
            "wheel": "sage_cu128_torch271_py39.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        },
        "cu128_torch280": {
            "torch_version": "2.8.0",
            "cuda_version": "12.8",
            "wheel": "sage_cu128_torch280_py39.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.8.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        },
        "cu130_torch290": {
            "torch_version": "2.9.0",
            "cuda_version": "13.0",
            "wheel": "sage_cu130_torch290_py39.whl",
            "python_folder": "3.9",
            "torch_install_cmd": "torch==2.9.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130"
        }
    },
    
    # Python 3.13 specific wheels
    "py313": {
        "cu130_torch290": {
            "torch_version": "2.9.0",
            "cuda_version": "13.0",
            "wheel": "sage_cu130_torch290_py313.whl",
            "python_folder": "3.13",
            "torch_install_cmd": "torch==2.9.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130"
        },
        "cu130_torch2100": {
            "torch_version": "2.10.0",
            "cuda_version": "13.0",
            "wheel": "sage_cu130_torch2100_py313.whl",
            "python_folder": "3.13",
            "torch_install_cmd": "torch==2.10.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130"
        }
    }
}

# Minimum required versions
MIN_PYTHON_VERSION = (3, 9)
MIN_TORCH_VERSION = "2.5.1"
MIN_CUDA_VERSION = "12.4"

# Latest recommended versions
RECOMMENDED_TORCH = "2.9.0"
RECOMMENDED_CUDA = "13.0"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SystemInfo:
    """System configuration information."""
    python_version: str
    python_tuple: Tuple[int, int, int]
    torch_version: Optional[str]
    cuda_version: Optional[str]
    sage_version: Optional[str]
    is_compatible: bool
    upgrade_needed: bool
    python_config_key: str  # "py39" or "py313"


@dataclass
class InstallationResult:
    """Installation process result."""
    success: bool
    previous_version: Optional[str]
    installed_version: Optional[str]
    errors: List[str]
    log_path: str


class InstallationStage(Enum):
    """Installation stages for logging and progress tracking."""
    INIT = "Initialization"
    SYSTEM_CHECK = "System Check"
    DOWNLOAD = "Download"
    INSTALL = "Installation"
    TORCH_UPGRADE = "PyTorch Upgrade"
    ROLLBACK = "Rollback"
    CLEANUP = "Cleanup"
    COMPLETE = "Complete"


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging() -> Tuple[logging.Logger, str]:
    """
    Setup dual logging system: console (INFO+) and file (DEBUG+).
    
    Returns:
        Tuple of (logger instance, log file path)
        
    Log filename format: TRSA_install_HH.MM-DD.MM.YYYY.log
    """
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%H.%M-%d.%m.%Y")
    log_filename = f"TRSA_install_{timestamp}.log"
    log_path = Path(log_filename).absolute()
    
    # Create logger
    logger = logging.getLogger('TRSAInstaller')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler (INFO and above, with colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler (DEBUG and above, detailed format)
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
    
    logger.addHandler(console_handler)
    
    return logger, str(log_path)
# ============================================================================
# CORE INSTALLER CLASS
# ============================================================================

class TRSAInstaller:
    """
    Main installer class with full lifecycle management.
    
    Handles:
    - Language selection and localization
    - System compatibility checks
    - PyTorch version detection and upgrade
    - Wheel selection based on Python/CUDA/Torch versions
    - Download and installation
    - Rollback on failure
    - Detailed logging
    """
    
    def __init__(self):
        """Initialize installer with logging and empty state."""
        self.logger, self.log_path = setup_logging()
        self.lang = 'en'
        self.system_info: Optional[SystemInfo] = None
        self.selected_config: Optional[Dict] = None
        self.errors: List[str] = []
        self.temp_files: List[Path] = []
        
        self.logger.info(f"TRSA Installer v{VERSION} initialized")
        self.logger.debug(f"Log file: {self.log_path}")
        self.logger.debug(f"Python: {sys.version}")
        self.logger.debug(f"Working directory: {Path.cwd()}")
    
    def t(self, key: str, **kwargs) -> str:
        """
        Shorthand for get_text with current language.
        
        Args:
            key: Translation key
            **kwargs: Format arguments
            
        Returns:
            Localized and formatted string
        """
        return get_text(self.lang, key, **kwargs)
    
    # ========================================================================
    # WELCOME & LANGUAGE SELECTION
    # ========================================================================
    
    def show_welcome_screen(self) -> None:
        """Display welcome banner and handle language selection."""
        print("\n")
        print(self.t("welcome_separator"))
        
        title = self.t("welcome_title")
        version_text = self.t("welcome_version", version=VERSION)
        
        # Center alignment
        separator_width = 70
        title_padding = (separator_width - len(title)) // 2
        version_padding = (separator_width - len(version_text)) // 2
        
        print(" " * title_padding + title)
        print(" " * version_padding + version_text)
        print(self.t("welcome_separator"))
        print()
        
        self.select_language()
        self.logger.info(f"Language selected: {self.lang}")
    
    def select_language(self) -> None:
        """
        Interactive language selection with system default fallback.
        
        Options:
        - 1: English
        - 2: Russian
        - Enter: System default
        """
        print(self.t("lang_select_prompt"))
        print(f"  {self.t('lang_option_en')}")
        print(f"  {self.t('lang_option_ru')}")
        print(f"  {self.t('lang_default')}")
        print()
        
        choice = input(self.t("lang_choice_prompt")).strip()
        
        if choice == '1':
            self.lang = 'en'
        elif choice == '2':
            self.lang = 'ru'
        elif choice == '':
            self.lang = get_system_language()
            self.logger.debug(f"Using system language: {self.lang}")
        else:
            print(self.t("lang_invalid"))
            self.lang = get_system_language()
            self.logger.warning(f"Invalid language choice '{choice}', using system default")
        
        print(self.t("lang_selected"))
        self.logger.info(f"Language set to: {self.lang}")
    
    # ========================================================================
    # SYSTEM CHECKS
    # ========================================================================
    
    def check_system(self) -> SystemInfo:
        """
        Comprehensive system compatibility check.
        
        Checks:
        - Python version (minimum 3.9)
        - PyTorch version and CUDA
        - Existing SageAttention installation
        - Compatibility and upgrade recommendations
        
        Returns:
            SystemInfo object with detected versions
            
        Raises:
            SystemExit: If Python version is too old or PyTorch not found
        """
        print(self.t("check_title"))
        print(self.t("welcome_separator"))
        
        self.logger.info("Starting system compatibility check")
        
        # Check Python version
        python_tuple = sys.version_info[:3]
        python_version = f"{python_tuple[0]}.{python_tuple[1]}.{python_tuple[2]}"
        
        print(self.t("check_python", version=python_version))
        self.logger.debug(f"Python: {python_version} ({sys.executable})")
        
        # Validate minimum Python version
        if python_tuple < MIN_PYTHON_VERSION:
            print(self.t("error_python_version", version=python_version))
            self.logger.error(f"Python {python_version} is below minimum {MIN_PYTHON_VERSION}")
            input(self.t("press_enter"))
            sys.exit(1)
        
        # Determine Python config key
        python_config_key = self._get_python_config_key(python_tuple)
        self.logger.debug(f"Python config key: {python_config_key}")
        
        # Check PyTorch and CUDA
        torch_version, cuda_version = self._get_torch_info()
        
        if torch_version:
            print(self.t("check_torch", version=torch_version))
            self.logger.debug(f"PyTorch: {torch_version}")
        else:
            print(self.t("error_torch_not_installed"))
            self.logger.error("PyTorch not installed")
            self.errors.append("PyTorch not found")
            input(self.t("press_enter"))
            sys.exit(1)
        
        if cuda_version:
            print(self.t("check_cuda", version=cuda_version))
            self.logger.debug(f"CUDA: {cuda_version}")
        else:
            self.logger.warning("CUDA version could not be determined")
        
        # Check SageAttention
        sage_version = self._get_sage_version()
        if sage_version:
            print(self.t("check_sage_installed", version=sage_version))
            self.logger.debug(f"SageAttention: {sage_version}")
        else:
            print(self.t("check_sage_not_installed"))
            self.logger.debug("SageAttention: Not installed")
        
        # Compatibility analysis
        is_compatible, upgrade_needed = self._check_compatibility(
            torch_version, cuda_version
        )
        
        system_info = SystemInfo(
            python_version=python_version,
            python_tuple=python_tuple,
            torch_version=torch_version,
            cuda_version=cuda_version,
            sage_version=sage_version,
            is_compatible=is_compatible,
            upgrade_needed=upgrade_needed,
            python_config_key=python_config_key
        )
        
        self.system_info = system_info
        print()
        
        if is_compatible and not upgrade_needed:
            print(self.t("check_compatible"))
            self.logger.info("System is fully compatible")
        elif upgrade_needed:
            print(self.t("check_upgrade_needed"))
            print(self.t("check_current_config", torch=torch_version, cuda=cuda_version or "N/A"))
            print(self.t("check_target_config", torch=RECOMMENDED_TORCH, cuda=RECOMMENDED_CUDA))
            self.logger.warning("System upgrade recommended")
        
        print(self.t("welcome_separator"))
        return system_info
    
    def _get_python_config_key(self, python_tuple: Tuple[int, int, int]) -> str:
        """
        Determine which configuration set to use based on Python version.
        
        Args:
            python_tuple: Python version as (major, minor, micro)
            
        Returns:
            'py313' for Python 3.13, 'py39' for 3.9-3.12
        """
        major, minor, _ = python_tuple
        
        if major == 3 and minor == 13:
            return "py313"
        elif major == 3 and minor >= 9:
            return "py39"
        else:
            self.logger.error(f"Unsupported Python version: {python_tuple}")
            return "py39"  # Fallback
    
    def _get_torch_info(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract PyTorch and CUDA versions from installed package.
        
        Returns:
            Tuple of (torch_version, cuda_version)
            
        Example:
            ("2.7.1", "12.8") from torch version "2.7.1+cu128"
        """
        try:
            import torch
            torch_version = torch.__version__
            
            self.logger.debug(f"Raw torch version: {torch_version}")
            
            # Parse version: "2.7.1+cu128" -> ("2.7.1", "12.8")
            match = re.match(r'(\d+\.\d+\.\d+)\+cu(\d+)', torch_version)
            if match:
                clean_torch = match.group(1)
                cuda_raw = match.group(2)
                
                # Convert "128" -> "12.8", "130" -> "13.0"
                if len(cuda_raw) == 3:
                    cuda_version = f"{cuda_raw[:-1]}.{cuda_raw[-1]}"
                elif len(cuda_raw) == 2:
                    cuda_version = f"{cuda_raw[0]}.{cuda_raw[1]}"
                else:
                    cuda_version = cuda_raw
                
                self.logger.debug(f"Parsed: torch={clean_torch}, cuda={cuda_version}")
                return clean_torch, cuda_version
            else:
                # No CUDA suffix (CPU-only build)
                clean_torch = torch_version.split('+')[0]
                self.logger.warning(f"No CUDA detected in torch version: {torch_version}")
                return clean_torch, None
                
        except ImportError:
            self.logger.error("PyTorch import failed")
            return None, None
        except Exception as e:
            self.logger.error(f"Error getting torch info: {e}", exc_info=True)
            return None, None
    
    def _get_sage_version(self) -> Optional[str]:
        """
        Get installed SageAttention version using pip.
        
        Returns:
            Version string or None if not installed
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "sageattention"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        self.logger.debug(f"SageAttention version: {version}")
                        return version
        except subprocess.TimeoutExpired:
            self.logger.warning("pip show timeout")
        except Exception as e:
            self.logger.debug(f"Could not get SageAttention version: {e}")
        
        return None
    
    def _check_compatibility(
        self, 
        torch_version: str, 
        cuda_version: Optional[str]
    ) -> Tuple[bool, bool]:
        """
        Check if current configuration is compatible and needs upgrade.
        
        Args:
            torch_version: Current PyTorch version
            cuda_version: Current CUDA version
            
        Returns:
            Tuple of (is_compatible, upgrade_needed)
            
        Logic:
        - Torch >= 2.9.0 + CUDA 13.0: Compatible, no upgrade
        - Torch >= 2.5.1: Compatible, upgrade recommended
        - Torch < 2.5.1: Not compatible, upgrade required
        """
        if not torch_version:
            return False, False
        
        # Already on recommended version
        if torch_version >= RECOMMENDED_TORCH and cuda_version == RECOMMENDED_CUDA:
            return True, False
        
        # Meets minimum requirements but could upgrade
        if torch_version >= MIN_TORCH_VERSION:
            if torch_version < RECOMMENDED_TORCH:
                return True, True
            return True, False
        
        # Below minimum - must upgrade
        return False, True
    
    # ========================================================================
    # PYTORCH UPGRADE
    # ========================================================================
    
    def prompt_torch_upgrade(self) -> bool:
        """
        Ask user if they want to upgrade PyTorch to recommended version.
        
        Returns:
            True if user approved upgrade
        """
        if not self.system_info or not self.system_info.upgrade_needed:
            return False
        
        print(self.t("torch_upgrade_title"))
        print(self.t("welcome_separator"))
        print(self.t("torch_upgrade_msg", 
                     current=self.system_info.torch_version,
                     cuda=self.system_info.cuda_version or "N/A"))
        
        # Recommend latest stable version
        latest_config = self._get_latest_config()
        if latest_config:
            print(self.t("torch_upgrade_recommend",
                         target=latest_config["torch_version"],
                         cuda_target=latest_config["cuda_version"]))
            print(self.t("torch_upgrade_benefits"))
        
        print()
        
        choice = input(self.t("torch_upgrade_prompt")).strip().lower()
        
        if choice in ['y', 'yes', 'д', 'да', '']:
            print(self.t("torch_upgrade_yes"))
            self.logger.info("User approved PyTorch upgrade")
            return True
        else:
            print(self.t("torch_upgrade_skip"))
            self.logger.info("User declined PyTorch upgrade")
            return False
    
    def _get_latest_config(self) -> Optional[Dict]:
        """
        Get latest recommended configuration for current Python version.
        
        Returns:
            Configuration dict or None
        """
        if not self.system_info:
            return None
        
        config_key = self.system_info.python_config_key
        configs = SUPPORTED_CONFIGS.get(config_key, {})
        
        # For py313, prefer latest torch
        if config_key == "py313" and "cu130_torch2100" in configs:
            return configs["cu130_torch2100"]
        
        # Otherwise use torch 2.9.0 + cu130
        if "cu130_torch290" in configs:
            return configs["cu130_torch290"]
        
        # Fallback to any cu130 config
        for key, config in configs.items():
            if config["cuda_version"] == "13.0":
                return config
        
        return None
    
    def upgrade_torch(self, config: Dict) -> bool:
        """
        Upgrade PyTorch to specified version.
        
        Args:
            config: Configuration dict with torch_install_cmd
            
        Returns:
            True if successful
        """
        print(self.t("install_torch_upgrading", version=config["torch_version"]))
        self.logger.info(f"Upgrading PyTorch to {config['torch_version']}")
        
        try:
            cmd = [
                sys.executable, "-m", "pip", "install", "--upgrade",
                *config["torch_install_cmd"].split()
            ]
            
            self.logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for large downloads
            )
            
            if result.returncode == 0:
                self.logger.info("PyTorch upgrade successful")
                # Update system info
                if self.system_info:
                    self.system_info.torch_version = config["torch_version"]
                    self.system_info.cuda_version = config["cuda_version"]
                return True
            else:
                error_msg = result.stderr[:300] if result.stderr else "Unknown error"
                self.logger.error(f"PyTorch upgrade failed: {error_msg}")
                self.errors.append(f"PyTorch upgrade failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("PyTorch upgrade timed out after 10 minutes")
            self.errors.append("PyTorch upgrade timeout")
            return False
        except Exception as e:
            self.logger.error(f"PyTorch upgrade exception: {e}", exc_info=True)
            self.errors.append(f"PyTorch upgrade error: {str(e)}")
            return False
    # ========================================================================
    # WHEEL SELECTION & INSTALLATION
    # ========================================================================
    
    def select_wheel_config(self) -> Optional[Dict]:
        """
        Select appropriate wheel configuration based on system info.
        
        Strategy:
        1. Try exact match for current Torch + CUDA + Python
        2. Try compatible match (nearest version)
        3. Fallback to py39 if py313 specific not available
        
        Returns:
            Configuration dict or None if no match found
        """
        if not self.system_info:
            return None
        
        torch_ver = self.system_info.torch_version
        cuda_ver = self.system_info.cuda_version
        python_key = self.system_info.python_config_key
        
        print(self.t("install_title"))
        print(self.t("welcome_separator"))
        print(self.t("install_selecting_wheel", 
                     torch=torch_ver, 
                     cuda=cuda_ver or "N/A",
                     python=self.system_info.python_version))
        
        self.logger.info(f"Selecting wheel: torch={torch_ver}, cuda={cuda_ver}, python_key={python_key}")
        
        # Get configs for Python version
        configs = SUPPORTED_CONFIGS.get(python_key, {})
        
        # Try exact match first
        for config_name, config in configs.items():
            if (self._version_match(torch_ver, config["torch_version"]) and 
                cuda_ver == config["cuda_version"]):
                print(self.t("install_wheel_found", wheel=config["wheel"]))
                self.logger.info(f"Exact match: {config_name}")
                self.selected_config = config
                return config
        
        # Try compatible match (same CUDA, compatible Torch)
        compatible = self._find_compatible_config(configs, torch_ver, cuda_ver)
        if compatible:
            print(self.t("install_wheel_found", wheel=compatible["wheel"]))
            self.logger.info(f"Compatible match: {compatible['wheel']}")
            self.selected_config = compatible
            return compatible
        
        # Fallback to py39 if we're on py313
        if python_key == "py313":
            self.logger.info("No py313 wheel found, trying py39 fallback")
            configs_py39 = SUPPORTED_CONFIGS.get("py39", {})
            
            for config_name, config in configs_py39.items():
                if (self._version_match(torch_ver, config["torch_version"]) and 
                    cuda_ver == config["cuda_version"]):
                    print(self.t("install_wheel_found", wheel=config["wheel"]))
                    self.logger.info(f"Py39 fallback match: {config_name}")
                    self.selected_config = config
                    return config
        
        # No match found
        print(self.t("install_wheel_not_found"))
        self.logger.error(f"No compatible wheel for torch={torch_ver}, cuda={cuda_ver}, python={python_key}")
        self.errors.append("No compatible wheel found")
        return None
    
    def _version_match(self, version1: str, version2: str) -> bool:
        """
        Check if two version strings match (major.minor level).
        
        Args:
            version1: First version string
            version2: Second version string
            
        Returns:
            True if versions match at major.minor level
        """
        try:
            v1_parts = version1.split('.')[:2]
            v2_parts = version2.split('.')[:2]
            return v1_parts == v2_parts
        except Exception:
            return version1 == version2
    
    def _find_compatible_config(
        self, 
        configs: Dict, 
        torch_ver: str, 
        cuda_ver: Optional[str]
    ) -> Optional[Dict]:
        """
        Find compatible config when exact match not available.
        
        Prefers higher torch versions with same CUDA.
        
        Args:
            configs: Available configurations
            torch_ver: Target torch version
            cuda_ver: Target CUDA version
            
        Returns:
            Best compatible config or None
        """
        if not cuda_ver:
            return None
        
        compatible_configs = []
        
        for config in configs.values():
            if config["cuda_version"] == cuda_ver:
                compatible_configs.append(config)
        
        if not compatible_configs:
            # Try same torch, different CUDA
            for config in configs.values():
                if self._version_match(torch_ver, config["torch_version"]):
                    compatible_configs.append(config)
        
        if compatible_configs:
            # Sort by torch version descending
            compatible_configs.sort(
                key=lambda x: x["torch_version"],
                reverse=True
            )
            return compatible_configs[0]
        
        return None
    
    def download_wheel(self, config: Dict) -> Optional[Path]:
        """
        Download wheel file from GitHub repository.
        
        Args:
            config: Configuration dict with wheel filename and folder
            
        Returns:
            Path to downloaded file or None on failure
        """
        wheel_name = config["wheel"]
        python_folder = config["python_folder"]
        url = f"{GITHUB_REPO}/{WHEELS_BASE_PATH}/{python_folder}/{wheel_name}"
        local_path = Path(wheel_name)
        
        print(self.t("install_downloading", file=wheel_name))
        self.logger.info(f"Downloading from: {url}")
        
        try:
            # Download with progress (if possible)
            urllib.request.urlretrieve(url, local_path)
            
            if local_path.exists() and local_path.stat().st_size > 0:
                self.temp_files.append(local_path)
                self.logger.info(f"Downloaded successfully: {local_path} ({local_path.stat().st_size} bytes)")
                return local_path
            else:
                self.logger.error(f"Downloaded file is empty or missing")
                return None
                
        except urllib.error.HTTPError as e:
            self.logger.error(f"HTTP error {e.code}: {e.reason}")
            self.errors.append(f"Download failed: HTTP {e.code}")
            print(self.t("error_download_failed", file=wheel_name))
            return None
        except urllib.error.URLError as e:
            self.logger.error(f"Network error: {e.reason}")
            self.errors.append("Network connection failed")
            print(self.t("error_network"))
            return None
        except Exception as e:
            self.logger.error(f"Download failed: {e}", exc_info=True)
            self.errors.append(f"Download error: {str(e)}")
            print(self.t("error_download_failed", file=wheel_name))
            return None
    
    def install_sageattention(self, wheel_path: Path) -> bool:
        """
        Install SageAttention from downloaded wheel file.
        
        Args:
            wheel_path: Path to wheel file
            
        Returns:
            True if installation successful
        """
        print(self.t("install_installing"))
        self.logger.info(f"Installing from: {wheel_path}")
        
        try:
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--upgrade", "--force-reinstall",
                str(wheel_path)
            ]
            
            self.logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes
            )
            
            if result.returncode == 0:
                print(self.t("install_success"))
                self.logger.info("SageAttention installation successful")
                return True
            else:
                print(self.t("install_failed"))
                error_msg = result.stderr[:300] if result.stderr else "Unknown error"
                self.logger.error(f"Installation failed: {error_msg}")
                self.errors.append(f"pip install failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            print(self.t("install_failed"))
            self.logger.error("Installation timed out")
            self.errors.append("Installation timeout")
            return False
        except Exception as e:
            print(self.t("install_failed"))
            self.logger.error(f"Installation exception: {e}", exc_info=True)
            self.errors.append(f"Installation error: {str(e)}")
            return False
    # ========================================================================
    # ROLLBACK
    # ========================================================================
    
    def prompt_rollback(self) -> bool:
        """
        Ask user if they want to rollback after failed installation.
        
        Returns:
            True if user wants rollback
        """
        if not self.system_info or not self.system_info.sage_version:
            self.logger.debug("No previous SageAttention version to rollback to")
            return False
        
        print(self.t("rollback_title"))
        print(self.t("welcome_separator"))
        
        choice = input(self.t("rollback_prompt")).strip().lower()
        
        if choice in ['y', 'yes', 'д', 'да', '']:
            self.logger.info("User approved rollback")
            return True
        else:
            print(self.t("rollback_skipped"))
            self.logger.info("User declined rollback")
            return False
    
    def rollback_sageattention(self) -> bool:
        """
        Rollback to previously installed SageAttention version.
        
        Returns:
            True if rollback successful
        """
        if not self.system_info or not self.system_info.sage_version:
            return False
        
        previous_version = self.system_info.sage_version
        
        print(self.t("rollback_starting"))
        self.logger.info(f"Rolling back to SageAttention v{previous_version}")
        
        try:
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--force-reinstall",
                f"sageattention=={previous_version}"
            ]
            
            self.logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode == 0:
                print(self.t("rollback_success"))
                self.logger.info("Rollback successful")
                return True
            else:
                print(self.t("rollback_failed"))
                error_msg = result.stderr[:300] if result.stderr else "Unknown error"
                self.logger.error(f"Rollback failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            print(self.t("rollback_failed"))
            self.logger.error("Rollback timed out")
            return False
        except Exception as e:
            print(self.t("rollback_failed"))
            self.logger.error(f"Rollback exception: {e}", exc_info=True)
            return False
    
    # ========================================================================
    # CLEANUP & SUMMARY
    # ========================================================================
    
    def cleanup(self) -> None:
        """Remove all temporary files created during installation."""
        print(self.t("cleanup_title"))
        print(self.t("cleanup_removing"))
        
        self.logger.info("Starting cleanup")
        
        removed_count = 0
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    self.logger.debug(f"Removed: {temp_file}")
                    removed_count += 1
            except Exception as e:
                self.logger.warning(f"Could not remove {temp_file}: {e}")
        
        print(self.t("cleanup_success"))
        self.logger.info(f"Cleanup completed ({removed_count} files removed)")
    
    def show_summary(self, success: bool) -> None:
        """
        Display comprehensive installation summary.
        
        Args:
            success: Whether installation was successful
        """
        print(self.t("summary_title"))
        print(self.t("welcome_separator"))
        
        # Success/failure header
        if success:
            print(self.t("summary_success"))
        else:
            print(self.t("summary_failed"))
        
        print()
        
        # Version information
        if self.system_info:
            if self.system_info.sage_version:
                print(self.t("summary_previous_version", 
                            version=self.system_info.sage_version))
            
            # Get newly installed version
            new_version = self._get_sage_version()
            if new_version:
                print(self.t("summary_installed_version", version=new_version))
            
            print()
            print(self.t("summary_python_version", 
                        version=self.system_info.python_version))
            print(self.t("summary_torch_version", 
                        version=self.system_info.torch_version or "N/A"))
            print(self.t("summary_cuda_version", 
                        version=self.system_info.cuda_version or "N/A"))
        
        # Error summary
        if self.errors:
            print()
            print(self.t("summary_errors", count=len(self.errors)))
            for i, error in enumerate(self.errors, 1):
                # Truncate long errors
                error_display = error[:150] + "..." if len(error) > 150 else error
                print(f"  {i}. {error_display}")
        
        # Log file location
        print()
        print(self.t("summary_log_saved", path=self.log_path))
        
        # Next steps (only on success)
        if success:
            print(self.t("summary_next_steps"))
            print(f"  {self.t('summary_next_step_1')}")
            print(f"  {self.t('summary_next_step_2')}")
            print(f"  {self.t('summary_next_step_3')}")
        
        print(self.t("welcome_separator"))
        
        self.logger.info(f"Installation {'successful' if success else 'failed'}")
        self.logger.info(f"Total errors: {len(self.errors)}")
    
    # ========================================================================
    # MAIN WORKFLOW
    # ========================================================================
    
    def run(self) -> InstallationResult:
        """
        Execute complete installation workflow.
        
        Workflow:
        1. Welcome screen & language selection
        2. System compatibility check
        3. PyTorch upgrade (if needed and approved)
        4. Wheel selection
        5. Download wheel
        6. Install SageAttention
        7. Rollback on failure (if approved)
        8. Cleanup temporary files
        9. Display summary
        
        Returns:
            InstallationResult with outcome details
        """
        try:
            # Stage 1: Welcome
            self.logger.info("=== Stage 1: Welcome ===")
            self.show_welcome_screen()
            
            # Stage 2: System Check
            self.logger.info("=== Stage 2: System Check ===")
            system_info = self.check_system()
            previous_sage = system_info.sage_version
            
            # Stage 3: PyTorch Upgrade (optional)
            if system_info.upgrade_needed:
                self.logger.info("=== Stage 3: PyTorch Upgrade ===")
                if self.prompt_torch_upgrade():
                    latest_config = self._get_latest_config()
                    if latest_config:
                        upgrade_success = self.upgrade_torch(latest_config)
                        if not upgrade_success:
                            self.logger.warning("PyTorch upgrade failed, continuing with current version")
                    else:
                        self.logger.warning("Could not determine latest config for upgrade")
            
            # Stage 4: Wheel Selection
            self.logger.info("=== Stage 4: Wheel Selection ===")
            config = self.select_wheel_config()
            if not config:
                print(self.t("install_wheel_not_found"))
                self.show_summary(False)
                return self._create_result(False, previous_sage, None)
            
            # Stage 5: Download
            self.logger.info("=== Stage 5: Download ===")
            wheel_path = self.download_wheel(config)
            if not wheel_path:
                self.show_summary(False)
                return self._create_result(False, previous_sage, None)
            
            # Stage 6: Installation
            self.logger.info("=== Stage 6: Installation ===")
            install_success = self.install_sageattention(wheel_path)
            
            # Stage 7: Rollback (if needed)
            if not install_success:
                self.logger.info("=== Stage 7: Rollback ===")
                if self.prompt_rollback():
                    self.rollback_sageattention()
                self.cleanup()
                self.show_summary(False)
                return self._create_result(False, previous_sage, None)
            
            # Stage 8: Get new version
            new_version = self._get_sage_version()
            
            # Stage 9: Cleanup
            self.logger.info("=== Stage 8: Cleanup ===")
            self.cleanup()
            
            # Stage 10: Summary
            self.logger.info("=== Stage 9: Summary ===")
            self.show_summary(True)
            
            return self._create_result(True, previous_sage, new_version)
            
        except KeyboardInterrupt:
            self.logger.warning("Installation cancelled by user (Ctrl+C)")
            print("\n\n" + self.t("summary_failed"))
            print("Installation cancelled by user.")
            self.cleanup()
            return self._create_result(False, None, None)
            
        except Exception as e:
            self.logger.critical(f"Unexpected error: {e}", exc_info=True)
            self.errors.append(f"Critical error: {str(e)}")
            self.cleanup()
            self.show_summary(False)
            return self._create_result(False, None, None)
    
    def _create_result(
        self, 
        success: bool, 
        previous: Optional[str], 
        installed: Optional[str]
    ) -> InstallationResult:
        """
        Create InstallationResult object.
        
        Args:
            success: Installation success status
            previous: Previous SageAttention version
            installed: Newly installed version
            
        Returns:
            InstallationResult object
        """
        return InstallationResult(
            success=success,
            previous_version=previous,
            installed_version=installed,
            errors=self.errors.copy(),
            log_path=self.log_path
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the installer.
    
    Creates installer instance, runs workflow, and exits with appropriate code.
    Exit codes:
        0 - Success
        1 - Failure
    """
    try:
        installer = TRSAInstaller()
        result = installer.run()
        
        # Wait for user acknowledgment
        input(installer.t("press_enter"))
        
        # Exit with appropriate code
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        print("Please check the log file for details.")
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()

