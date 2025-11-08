#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI SageAttention Installer
Version: 2.6.0
Author: freyandere
Repository: https://github.com/freyandere/TRSA-Comfyui_installer

Production-ready installer with comprehensive error handling,
logging, and rollback capabilities.
"""

import sys
import os
import subprocess
import re
import logging
import urllib.request
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
    print("ERROR: installer_core_lang.py not found!")
    print("Please ensure both installer files are in the same directory.")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "2.6.0"
GITHUB_REPO = "https://raw.githubusercontent.com/freyandere/TRSA-Comfyui_installer/main"
WHEELS_FOLDER = "wheels"

# Supported configurations
SUPPORTED_CONFIGS = {
    "cu128_torch271": {
        "torch_version": "2.7.1",
        "cuda_version": "12.8",
        "wheel": "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl",
        "torch_install_cmd": "torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    },
    "cu128_torch280": {
        "torch_version": "2.8.0",
        "cuda_version": "12.8",
        "wheel": "sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl",
        "torch_install_cmd": "torch==2.8.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    },
    "cu130_torch290": {
        "torch_version": "2.9.0",
        "cuda_version": "13.0",
        "wheel": "sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
        "torch_install_cmd": "torch==2.9.0+cu130 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130"
    }
}

# Minimum required versions
MIN_TORCH_VERSION = "2.7.1"
MIN_CUDA_VERSION = "12.8"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SystemInfo:
    """System configuration information."""
    python_version: str
    torch_version: Optional[str]
    cuda_version: Optional[str]
    sage_version: Optional[str]
    is_compatible: bool
    upgrade_needed: bool


@dataclass
class InstallationResult:
    """Installation process result."""
    success: bool
    previous_version: Optional[str]
    installed_version: Optional[str]
    errors: List[str]
    log_path: str


class InstallationStage(Enum):
    """Installation stages for logging."""
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
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging() -> Tuple[logging.Logger, str]:
    """
    Setup dual logging: console + detailed file.
    
    Returns:
        Tuple of (logger, log_file_path)
    """
    # Generate log filename: HH.MM-DD.MM.YYYY.log
    timestamp = datetime.now().strftime("%H.%M-%d.%m.%Y")
    log_filename = f"TRSA_install_{timestamp}.log"
    log_path = Path(log_filename).absolute()
    
    # Create logger
    logger = logging.getLogger('TRSAInstaller')
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (DEBUG and above)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, str(log_path)


# ============================================================================
# CORE INSTALLER CLASS
# ============================================================================

class TRSAInstaller:
    """Main installer class with full lifecycle management."""
    
    def __init__(self):
        self.logger, self.log_path = setup_logging()
        self.lang = 'en'
        self.system_info: Optional[SystemInfo] = None
        self.selected_config: Optional[Dict] = None
        self.errors: List[str] = []
        self.temp_files: List[Path] = []
        
        self.logger.info(f"TRSA Installer v{VERSION} initialized")
        self.logger.debug(f"Log file: {self.log_path}")
    
    def t(self, key: str, **kwargs) -> str:
        """Shorthand for get_text with current language."""
        return get_text(self.lang, key, **kwargs)
    
    # ========================================================================
    # WELCOME & LANGUAGE SELECTION
    # ========================================================================
    
    def show_welcome_screen(self) -> None:
        """Display welcome screen and handle language selection."""
        print("\n")
        print(self.t("welcome_separator"))
        title = self.t("welcome_title")
        version_text = self.t("welcome_version", version=VERSION)
        
        # Center align
        title_padding = (60 - len(title)) // 2
        version_padding = (60 - len(version_text)) // 2
        
        print(" " * title_padding + title)
        print(" " * version_padding + version_text)
        print(self.t("welcome_separator"))
        print()
        
        self.select_language()
        self.logger.info(f"Language selected: {self.lang}")
    
    def select_language(self) -> None:
        """Handle language selection with fallback to system default."""
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
        
        print(self.t("lang_selected"))
        self.logger.info(f"Language set to: {self.lang}")
    
    # ========================================================================
    # SYSTEM CHECKS
    # ========================================================================
    
    def check_system(self) -> SystemInfo:
        """
        Comprehensive system compatibility check.
        
        Returns:
            SystemInfo object with all detected versions
        """
        print(self.t("check_title"))
        print(self.t("welcome_separator"))
        
        self.logger.info("Starting system compatibility check")
        
        # Check Python
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(self.t("check_python", version=python_version))
        self.logger.debug(f"Python: {python_version}")
        
        # Check PyTorch
        torch_version, cuda_version = self._get_torch_info()
        
        if torch_version:
            print(self.t("check_torch", version=torch_version))
            self.logger.debug(f"PyTorch: {torch_version}")
        else:
            print(self.t("error_torch_not_installed"))
            self.logger.error("PyTorch not installed")
            self.errors.append("PyTorch not found")
            sys.exit(1)
        
        if cuda_version:
            print(self.t("check_cuda", version=cuda_version))
            self.logger.debug(f"CUDA: {cuda_version}")
        
        # Check SageAttention
        sage_version = self._get_sage_version()
        if sage_version:
            print(self.t("check_sage_installed", version=sage_version))
            self.logger.debug(f"SageAttention: {sage_version}")
        else:
            print(self.t("check_sage_not_installed"))
            self.logger.debug("SageAttention: Not installed")
        
        # Compatibility check
        is_compatible, upgrade_needed = self._check_compatibility(torch_version, cuda_version)
        
        system_info = SystemInfo(
            python_version=python_version,
            torch_version=torch_version,
            cuda_version=cuda_version,
            sage_version=sage_version,
            is_compatible=is_compatible,
            upgrade_needed=upgrade_needed
        )
        
        self.system_info = system_info
        print()
        
        if is_compatible and not upgrade_needed:
            print(self.t("check_compatible"))
            self.logger.info("System is fully compatible")
        elif upgrade_needed:
            print(self.t("check_upgrade_needed"))
            print(self.t("check_current_config", torch=torch_version, cuda=cuda_version))
            self.logger.warning("System upgrade recommended")
        
        print(self.t("welcome_separator"))
        return system_info
    
    def _get_torch_info(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract PyTorch and CUDA versions.
        
        Returns:
            Tuple of (torch_version, cuda_version)
        """
        try:
            import torch
            torch_version = torch.__version__
            
            # Parse version (e.g., "2.7.1+cu128" -> "2.7.1", "12.8")
            match = re.match(r'(\d+\.\d+\.\d+)\+cu(\d+)', torch_version)
            if match:
                clean_torch = match.group(1)
                cuda_raw = match.group(2)
                # Convert "128" -> "12.8", "130" -> "13.0"
                cuda_version = f"{cuda_raw[:-1]}.{cuda_raw[-1]}"
                return clean_torch, cuda_version
            else:
                # No CUDA suffix
                return torch_version.split('+')[0], None
        except ImportError:
            return None, None
        except Exception as e:
            self.logger.error(f"Error getting torch info: {e}")
            return None, None
    
    def _get_sage_version(self) -> Optional[str]:
        """
        Get installed SageAttention version.
        
        Returns:
            Version string or None
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "sageattention"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':', 1)[1].strip()
        except Exception as e:
            self.logger.debug(f"Could not get SageAttention version: {e}")
        
        return None
    
    def _check_compatibility(self, torch_version: str, cuda_version: Optional[str]) -> Tuple[bool, bool]:
        """
        Check if current configuration is compatible.
        
        Args:
            torch_version: Current PyTorch version
            cuda_version: Current CUDA version
        
        Returns:
            Tuple of (is_compatible, upgrade_needed)
        """
        if not torch_version:
            return False, False
        
        # Check if already on latest (2.9.0 + cu130)
        if torch_version >= "2.9.0" and cuda_version == "13.0":
            return True, False
        
        # Check if meets minimum requirements
        if torch_version >= MIN_TORCH_VERSION:
            # Compatible but upgrade recommended
            if torch_version < "2.9.0":
                return True, True
            return True, False
        
        # Below minimum - upgrade required
        return False, True
    
    # ========================================================================
    # TORCH UPGRADE
    # ========================================================================
    
    def prompt_torch_upgrade(self) -> bool:
        """
        Ask user if they want to upgrade PyTorch.
        
        Returns:
            True if user wants to upgrade
        """
        if not self.system_info or not self.system_info.upgrade_needed:
            return False
        
        print(self.t("torch_upgrade_title"))
        print(self.t("welcome_separator"))
        print(self.t("torch_upgrade_msg", 
                     current=self.system_info.torch_version,
                     cuda=self.system_info.cuda_version or "N/A"))
        
        # Recommend latest
        latest_config = SUPPORTED_CONFIGS["cu130_torch290"]
        print(self.t("torch_upgrade_recommend",
                     target=latest_config["torch_version"],
                     cuda_target=latest_config["cuda_version"]))
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
    
    def upgrade_torch(self) -> bool:
        """
        Upgrade PyTorch to latest supported version.
        
        Returns:
            True if successful
        """
        latest_config = SUPPORTED_CONFIGS["cu130_torch290"]
        
        print(self.t("install_torch_upgrading", version=latest_config["torch_version"]))
        self.logger.info(f"Upgrading PyTorch to {latest_config['torch_version']}")
        
        try:
            cmd = [
                sys.executable, "-m", "pip", "install", "--upgrade",
                *latest_config["torch_install_cmd"].split()
            ]
            
            self.logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            if result.returncode == 0:
                self.logger.info("PyTorch upgrade successful")
                # Update system info
                self.system_info.torch_version = latest_config["torch_version"]
                self.system_info.cuda_version = latest_config["cuda_version"]
                return True
            else:
                self.logger.error(f"PyTorch upgrade failed: {result.stderr}")
                self.errors.append(f"PyTorch upgrade failed: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("PyTorch upgrade timed out")
            self.errors.append("PyTorch upgrade timeout")
            return False
        except Exception as e:
            self.logger.error(f"PyTorch upgrade exception: {e}")
            self.errors.append(f"PyTorch upgrade error: {str(e)}")
            return False
    
    # ========================================================================
    # WHEEL SELECTION & INSTALLATION
    # ========================================================================
    
    def select_wheel_config(self) -> Optional[Dict]:
        """
        Select appropriate wheel configuration based on system.
        
        Returns:
            Configuration dict or None
        """
        if not self.system_info:
            return None
        
        torch_ver = self.system_info.torch_version
        cuda_ver = self.system_info.cuda_version
        
        print(self.t("install_title"))
        print(self.t("welcome_separator"))
        print(self.t("install_selecting_wheel", torch=torch_ver, cuda=cuda_ver))
        
        # Match configuration
        for config_name, config in SUPPORTED_CONFIGS.items():
            if (torch_ver.startswith(config["torch_version"]) and 
                cuda_ver == config["cuda_version"]):
                print(self.t("install_wheel_found", wheel=config["wheel"]))
                self.logger.info(f"Selected config: {config_name}")
                self.selected_config = config
                return config
        
        # No exact match - use highest compatible
        if torch_ver >= "2.9.0":
            config = SUPPORTED_CONFIGS["cu130_torch290"]
        elif torch_ver >= "2.8.0":
            config = SUPPORTED_CONFIGS["cu128_torch280"]
        else:
            config = SUPPORTED_CONFIGS["cu128_torch271"]
        
        print(self.t("install_wheel_found", wheel=config["wheel"]))
        self.logger.info(f"Using compatible config: {config['wheel']}")
        self.selected_config = config
        return config
    
    def download_wheel(self, wheel_name: str) -> Optional[Path]:
        """
        Download wheel file from GitHub repository.
        
        Args:
            wheel_name: Name of the wheel file
        
        Returns:
            Path to downloaded file or None
        """
        url = f"{GITHUB_REPO}/{WHEELS_FOLDER}/{wheel_name}"
        local_path = Path(wheel_name)
        
        print(self.t("install_downloading", file=wheel_name))
        self.logger.info(f"Downloading from: {url}")
        
        try:
            urllib.request.urlretrieve(url, local_path)
            self.temp_files.append(local_path)
            self.logger.info(f"Downloaded to: {local_path}")
            return local_path
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            self.errors.append(f"Download failed: {wheel_name}")
            print(self.t("error_download_failed", file=wheel_name))
            return None
    
    def install_sageattention(self, wheel_path: Path) -> bool:
        """
        Install SageAttention from wheel file.
        
        Args:
            wheel_path: Path to wheel file
        
        Returns:
            True if successful
        """
        print(self.t("install_installing"))
        self.logger.info(f"Installing from: {wheel_path}")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", str(wheel_path)]
            
            self.logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(self.t("install_success"))
                self.logger.info("SageAttention installation successful")
                return True
            else:
                print(self.t("install_failed"))
                self.logger.error(f"Installation failed: {result.stderr}")
                self.errors.append(f"Installation failed: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Installation timed out")
            self.errors.append("Installation timeout")
            return False
        except Exception as e:
            self.logger.error(f"Installation exception: {e}")
            self.errors.append(f"Installation error: {str(e)}")
            return False
    
    # ========================================================================
    # ROLLBACK
    # ========================================================================
    
    def prompt_rollback(self) -> bool:
        """
        Ask user if they want to rollback.
        
        Returns:
            True if user wants rollback
        """
        if not self.system_info or not self.system_info.sage_version:
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
        Rollback to previously installed version.
        
        Returns:
            True if successful
        """
        if not self.system_info or not self.system_info.sage_version:
            return False
        
        previous_version = self.system_info.sage_version
        
        print(self.t("rollback_starting"))
        self.logger.info(f"Rolling back to v{previous_version}")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", f"sageattention=={previous_version}"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(self.t("rollback_success"))
                self.logger.info("Rollback successful")
                return True
            else:
                print(self.t("rollback_failed"))
                self.logger.error(f"Rollback failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(self.t("rollback_failed"))
            self.logger.error(f"Rollback exception: {e}")
            return False
    
    # ========================================================================
    # CLEANUP & SUMMARY
    # ========================================================================
    
    def cleanup(self) -> None:
        """Remove temporary files."""
        print(self.t("cleanup_title"))
        print(self.t("cleanup_removing"))
        
        self.logger.info("Starting cleanup")
        
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    self.logger.debug(f"Removed: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Could not remove {temp_file}: {e}")
        
        print(self.t("cleanup_success"))
        self.logger.info("Cleanup completed")
    
    def show_summary(self, success: bool) -> None:
        """
        Display installation summary.
        
        Args:
            success: Whether installation was successful
        """
        print(self.t("summary_title"))
        print(self.t("welcome_separator"))
        
        if success:
            print(self.t("summary_success"))
        else:
            print(self.t("summary_failed"))
        
        if self.system_info:
            if self.system_info.sage_version:
                print(self.t("summary_previous_version", version=self.system_info.sage_version))
            
            # Get new version
            new_version = self._get_sage_version()
            if new_version:
                print(self.t("summary_installed_version", version=new_version))
            
            print(self.t("summary_torch_version", version=self.system_info.torch_version))
            print(self.t("summary_cuda_version", version=self.system_info.cuda_version or "N/A"))
        
        if self.errors:
            print(self.t("summary_errors", count=len(self.errors)))
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error[:100]}")
        
        print(self.t("summary_log_saved", path=self.log_path))
        print(self.t("welcome_separator"))
        
        self.logger.info(f"Installation {'successful' if success else 'failed'}")
    
    # ========================================================================
    # MAIN WORKFLOW
    # ========================================================================
    
    def run(self) -> InstallationResult:
        """
        Execute complete installation workflow.
        
        Returns:
            InstallationResult with outcome details
        """
        try:
            # 1. Welcome & Language
            self.show_welcome_screen()
            
            # 2. System Check
            system_info = self.check_system()
            previous_sage = system_info.sage_version
            
            # 3. PyTorch Upgrade (if needed)
            if system_info.upgrade_needed:
                if self.prompt_torch_upgrade():
                    if not self.upgrade_torch():
                        self.logger.warning("PyTorch upgrade failed, continuing with current version")
            
            # 4. Select Wheel
            config = self.select_wheel_config()
            if not config:
                print(self.t("install_wheel_not_found"))
                self.errors.append("No compatible wheel found")
                return self._create_result(False, previous_sage, None)
            
            # 5. Download Wheel
            wheel_path = self.download_wheel(config["wheel"])
            if not wheel_path:
                return self._create_result(False, previous_sage, None)
            
            # 6. Install SageAttention
            install_success = self.install_sageattention(wheel_path)
            
            # 7. Rollback if failed
            if not install_success:
                if self.prompt_rollback():
                    self.rollback_sageattention()
                return self._create_result(False, previous_sage, None)
            
            # 8. Get new version
            new_version = self._get_sage_version()
            
            # 9. Cleanup
            self.cleanup()
            
            # 10. Summary
            self.show_summary(True)
            
            return self._create_result(True, previous_sage, new_version)
            
        except KeyboardInterrupt:
            self.logger.warning("Installation cancelled by user")
            print("\n\nInstallation cancelled.")
            return self._create_result(False, None, None)
        except Exception as e:
            self.logger.critical(f"Unexpected error: {e}", exc_info=True)
            self.errors.append(f"Critical error: {str(e)}")
            self.show_summary(False)
            return self._create_result(False, None, None)
    
    def _create_result(self, success: bool, previous: Optional[str], installed: Optional[str]) -> InstallationResult:
        """Create InstallationResult object."""
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
    """Main entry point."""
    installer = TRSAInstaller()
    result = installer.run()
    
    # Wait for user
    input(installer.t("press_enter"))
    
    # Exit code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
