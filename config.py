#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Accelerator v5.1 - Configuration
Централизованное хранение констант и настроек
"""
from typing import Dict, Any


class Config:
    """Централизованная конфигурация приложения"""
    
    # Версии и требования
    APP_VERSION = "5.1"
    MIN_PYTHON_VERSION = "3.12"
    TARGET_PYTORCH_VERSION = "2.7.1"
    MAX_PYTORCH_VERSION = "2.7.9"
    
    # URLs и пути
    PYTORCH_INSTALL_URL = "https://download.pytorch.org/whl/cu128"
    SAGEATTENTION_WHEEL_URL = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
    INCLUDE_LIBS_URL = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/python_3.12.7_include_libs.zip"
    
    # Файлы
    SAGEATTENTION_WHEEL_FILE = "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
    INCLUDE_LIBS_FILE = "include_libs.zip"
    
    # Таймауты
    COMMAND_TIMEOUT_DEFAULT = 300
    PYTORCH_INSTALL_TIMEOUT = 900
    REGISTRY_TIMEOUT = 5
    
    # Локали для автоопределения языка
    LOCALE_ENV_VARS = ['LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE']
    REGISTRY_LOCALE_PATH = 'HKCU\\Control Panel\\International'
    
    @classmethod
    def get_python_executables(cls) -> list:
        """Список возможных имен Python исполняемых файлов"""
        return ["python.exe", "python"]
    
    @classmethod
    def get_pytorch_packages(cls) -> list:
        """Список пакетов PyTorch для установки"""
        return [
            f"torch=={cls.TARGET_PYTORCH_VERSION}",
            "torchvision",
            "torchaudio"
        ]
    
    @classmethod
    def get_pytorch_install_args(cls) -> list:
        """Аргументы для установки PyTorch"""
        return [
            "--force-reinstall",
            "--index-url",
            cls.PYTORCH_INSTALL_URL
        ]
