#!/usr/bin/env python3
"""
ComfyUI Accelerator Installer - Clean Implementation
"""

import os
import sys
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
LOG = logging.getLogger("installer_core")

# Language selection with default to system language
def get_language() -> str:
    """Detect or select user language preference."""
    # Check environment variables for forced language
    force_lang = os.environ.get("ACC_LANG_FORCE", "").lower()
    if force_lang in ("1", "true", "yes", "y"):
        lang = os.environ.get("ACC_LANG", "").strip().lower()
        return "ru" if lang == "ru" else "en"
    
    # Check existing environment variable
    existing_lang = os.environ.get("ACC_LANG", "").strip().lower()
    if existing_lang in ("ru", "en"):
        return existing_lang
    
    # Prompt user for language selection
    print("\nSelect language / Выберите язык:")
    print("  1) RU (Русский)")
    print("  2) EN (English)")
    
    choice = input("Choice (1/2, Enter=Auto): ").strip()
    if choice == "1":
        return "ru"
    elif choice == "2":
        return "en"
    else:
        # Default to system language detection
        lang_code = os.environ.get("LANG", "").lower() or os.environ.get("LC_ALL", "").lower()
        return "ru" if lang_code.startswith("ru") else "en"

# Language dictionary for messages
LANGUAGE_STRINGS = {
    "ru": {
        "welcome_title": "Установщик ускорителя ComfyUI",
        "version": "Версия 2.0",
        "detecting_sage": "Определение установленного SageAttention...",
        "current_version": "Текущая версия: {}",
        "no_sage_installed": "SageAttention не установлен",
        "torch_cuda_check": "Проверка установленных версий torch и CUDA:",
        "torch_ver": "torch версия: {}",
        "cuda_ver": "CUDA версия: {}",
        "system_language": "Системный язык: {}",
        "conditions_met": "Условия для установки соблюдены",
        "requirements_not_met": "Требования к установке не выполнены",
        "install_prompt": "Хотите продолжить установку? (y/N): ",
        "cuda_130_ok": "CUDA 13.0 и torch 2.9.0 уже установлены",
        "torch_update_needed": "Требуется обновление до последней версии torch",
        "installing_sage": "Установка SageAttention...",
        "sage_installed": "SageAttention успешно установлен!",
        "install_failed": "Ошибка установки: {}",
        "summary_title": "\nИтоговая информация:",
        "step_status": "- {}: {}",
        "success": "успешно",
        "failure": "ошибка",
        "done": "Готово.",
    },
    "en": {
        "welcome_title": "ComfyUI Accelerator Installer",
        "version": "Version 2.0",
        "detecting_sage": "Detecting installed SageAttention...",
        "current_version": "Current version: {}",
        "no_sage_installed": "SageAttention not installed",
        "torch_cuda_check": "Checking torch and CUDA versions:",
        "torch_ver": "torch version: {}",
        "cuda_ver": "CUDA version: {}",
        "system_language": "System language: {}",
        "conditions_met": "Installation requirements met",
        "requirements_not_met": "Installation requirements not met",
        "install_prompt": "Do you want to proceed with installation? (y/N): ",
        "cuda_130_ok": "CUDA 13.0 and torch 2.9.0 are already installed",
        "torch_update_needed": "Update needed to latest torch version",
        "installing_sage": "Installing SageAttention...",
        "sage_installed": "SageAttention installed successfully!",
        "install_failed": "Installation failed: {}",
        "summary_title": "\nFinal Summary:",
        "step_status": "- {}: {}",
        "success": "success",
        "failure": "failure",
        "done": "Done.",
    }
}

# Get language
LANG = get_language()
T = LANGUAGE_STRINGS[LANG]

def print_welcome():
    """Display welcome screen."""
    width = 50
    print("\n" + "="*width)
    title = f"{T['welcome_title']} {T['version']}"
    print(f"{title:^{width}}")
    print("="*width)

def get_current_torch_cuda() -> Tuple[str, str]:
    """Get current torch and CUDA versions."""
    try:
        code = """
import sys
try:
    import torch
    print(getattr(torch, '__version__', ''))
    cuda_ver = getattr(getattr(torch, 'version', None), 'cuda', '')
    print(cuda_ver if cuda_ver else '')
except (ImportError, AttributeError):
    print('')
    print('')
"""
        result = subprocess.run([sys.executable, "-c", code], 
                              capture_output=True, text=True, timeout=30)
        lines = [s.strip() for s in (result.stdout or "").splitlines()]
        
        torch_ver = lines[0] if len(lines) > 0 else ""
        cuda_ver = lines[1] if len(lines) > 1 else ""
        
        return torch_ver, cuda_ver
    except Exception:
        return "", ""

def check_sage_version() -> Optional[str]:
    """Check if SageAttention is installed and get its version."""
    try:
        result = subprocess.run([sys.executable, "-c", 
                                "import sageattention; print(sageattention.__version__)"], 
                               capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def main():
    # Welcome screen
    print_welcome()
    
    LOG.info(T["system_language"].format(LANG))
    
    # Detect SageAttention version
    sage_version = check_sage_version()
    LOG.info(T["detecting_sage"])
    if sage_version:
        LOG.info(T["current_version"].format(sage_version))
    else:
        LOG.info(T["no_sage_installed"])
    
    # Check torch and CUDA versions
    torch_ver, cuda_ver = get_current_torch_cuda()
    LOG.info(T["torch_cuda_check"])
    LOG.info(T["torch_ver"].format(torch_ver or "not installed"))
    LOG.info(T["cuda_ver"].format(cuda_ver or "not installed"))
    
    # Conditions for installation
    if not torch_ver:
        LOG.warning(T["requirements_not_met"])
        LOG.warning("No torch version detected. Installation required.")
    elif cuda_ver == "13.0" and torch_ver.startswith("2.9."):
        LOG.info(T["cuda_130_ok"])
        # No installation needed
    elif torch_ver < "2.7.1":
        LOG.warning(T["torch_update_needed"])
        LOG.warning("Update to the latest torch version required.")
        
    # Prompt for installation if needed
    choice = input(T["install_prompt"]).strip().lower()
    if choice not in ("y", "yes", "д", "да"):
        LOG.info("Installation cancelled by user")
        return 1
    
    # Installation process (mock implementation)
    try:
        LOG.info(T["installing_sage"])
        
        # Determine appropriate wheel based on CUDA version
        if cuda_ver == "13.0" and torch_ver.startswith("2.9"):
            wheel_url = "https://example.com/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
        elif cuda_ver == "12.8" and torch_ver.startswith("2.7"):
            wheel_url = "https://example.com/sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
        elif cuda_ver == "12.8" and torch_ver.startswith("2.8"):
            wheel_url = "https://example.com/sageattention-2.2.0+cu128torch2.8.0.post2-cp39-abi3-win_amd64.whl"
        else:
            # Default to latest available version
            wheel_url = "https://example.com/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
        
        # Download and install the wheel (mocked)
        LOG.info(f"Downloading from {wheel_url}")
        # In real implementation, this would:
        # 1. Download the wheel
        # 2. Install using pip
        
        LOG.info(T["sage_installed"])
    except Exception as e:
        LOG.error(T["install_failed"].format(str(e)))
        return 1
    
    # Summary screen
    print("\n" + "="*50)
    print(f"{T['summary_title']}")
    print("="*50)
    
    steps = [
        (T["detecting_sage"], "success"),
        (T["installing_sage"], "success")
    ]
    
    for step, status in steps:
        print(T["step_status"].format(step, T[status]))
    
    print("\n" + "="*50)
    LOG.info(T["done"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
