#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Accelerator v5.1 - Main Logic (Temporary Module)
Основная логика установки для bat-контекста с модульной архитектурой
"""
import subprocess
import urllib.request
import zipfile
import sys
import os
import warnings
import importlib.util
from pathlib import Path
from typing import Tuple, Optional, Literal, List

# === АДАПТИРОВАННЫЙ ИМПОРТ ДЛЯ BAT-КОНТЕКСТА ===

def import_temp_module(module_name: str, file_name: str):
    """
    Безопасный импорт временных модулей в bat-контексте
    
    Args:
        module_name: имя модуля для импорта
        file_name: имя файла модуля
        
    Returns:
        Импортированный модуль или None
    """
    try:
        # Попытка импорта как временного модуля
        if file_name in sys.modules:
            return sys.modules[file_name]
        
        if os.path.exists(file_name):
            spec = importlib.util.spec_from_file_location(module_name, file_name)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[file_name] = module
                spec.loader.exec_module(module)
                return module
        
        # Fallback - прямой импорт
        return __import__(module_name)
        
    except Exception as e:
        print(f"Warning: Failed to import {module_name}: {e}")
        return None

# Импортируем модули с обработкой bat-контекста
config_module = import_temp_module('config', 'temp_config.py')
ui_module = import_temp_module('ui', 'temp_ui.py')

if config_module:
    Config = config_module.Config
else:
    # Минимальная конфигурация для fallback
    class Config:
        APP_VERSION = "5.1"
        MIN_PYTHON_VERSION = "3.12"
        TARGET_PYTORCH_VERSION = "2.7.1"
        MAX_PYTORCH_VERSION = "2.7.9"
        PYTORCH_INSTALL_URL = "https://download.pytorch.org/whl/cu128"
        SAGEATTENTION_WHEEL_URL = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
        INCLUDE_LIBS_URL = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/python_3.12.7_include_libs.zip"
        SAGEATTENTION_WHEEL_FILE = "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
        INCLUDE_LIBS_FILE = "include_libs.zip"
        COMMAND_TIMEOUT_DEFAULT = 300
        PYTORCH_INSTALL_TIMEOUT = 900
        
        @classmethod
        def get_python_executables(cls) -> List[str]:
            return ["./python.exe", "python.exe", "python"]
        
        @classmethod
        def get_pytorch_packages(cls) -> List[str]:
            return [f"torch=={cls.TARGET_PYTORCH_VERSION}", "torchvision", "torchaudio"]
        
        @classmethod
        def get_pytorch_install_args(cls) -> List[str]:
            return ["--force-reinstall", "--index-url", cls.PYTORCH_INSTALL_URL]

if ui_module:
    LocalizationManager = ui_module.LocalizationManager
    UserInterface = ui_module.UserInterface
else:
    # Минимальная UI для fallback
    class LocalizationManager:
        def __init__(self):
            self.language = 'en'
        def get_message(self, key: str, **kwargs) -> str:
            return key
        def ask_language_choice(self) -> str:
            return 'en'
    
    class UserInterface:
        def __init__(self, loc):
            self.loc = loc
        def print_header(self): print("TRSA ComfyUI Accelerator")
        def print_step(self, step, total, key): print(f"{step}/{total} {key}")
        def print_success(self, msg): print(f"✅ {msg}")
        def print_error(self, msg): print(f"❌ {msg}")
        def print_warning(self, msg): print(f"⚠️ {msg}")
        def ask_pytorch_action(self, version, action): return True
        def print_final_result(self, success): print("Installation completed")
        def wait_for_exit(self): pass


# === ОСНОВНЫЕ КЛАССЫ ЛОГИКИ ===

class VersionManager:
    """Менеджер сравнения и управления версиями"""
    
    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """Интеллектуальное сравнение версий PyTorch"""
        def clean_version(version: str) -> list:
            try:
                clean_str = version.split('+')[0]
                return [int(x) for x in clean_str.split('.')]
            except (ValueError, AttributeError):
                return [0]
        
        try:
            v1_parts = clean_version(version1)
            v2_parts = clean_version(version2)
            
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0
            
        except Exception:
            return -1 if version1 < version2 else (1 if version1 > version2 else 0)
    
    @classmethod
    def get_pytorch_status(cls, current_version: str) -> Literal['too_old', 'compatible', 'too_new']:
        """Определение статуса версии PyTorch"""
        min_compare = cls.compare_versions(current_version, Config.TARGET_PYTORCH_VERSION)
        max_compare = cls.compare_versions(current_version, Config.MAX_PYTORCH_VERSION)
        
        if min_compare < 0:
            return 'too_old'
        elif max_compare > 0:
            return 'too_new'
        else:
            return 'compatible'


class SystemManager:
    """Менеджер системных операций для bat-контекста"""
    
    def __init__(self, localization: LocalizationManager):
        self.loc = localization
        self.python_exe = self._find_python_executable()
    
    def _find_python_executable(self) -> str:
        """Поиск исполняемого файла Python с приоритетом для bat-контекста"""
        for exe in Config.get_python_executables():
            if os.path.exists(exe):
                return exe
        
        raise FileNotFoundError("Python executable not found")
    
    def run_command(self, cmd: list, timeout: int = None) -> Tuple[bool, str]:
        """Выполнение системной команды с обработкой ошибок"""
        if timeout is None:
            timeout = Config.COMMAND_TIMEOUT_DEFAULT
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout, 
                encoding='utf-8'
            )
            return result.returncode == 0, result.stdout + result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "Command timeout"
        except Exception as e:
            return False, str(e)
    
    def download_file(self, url: str, filename: str) -> bool:
        """Безопасное скачивание файла"""
        try:
            urllib.request.urlretrieve(url, filename)
            return os.path.exists(filename) and os.path.getsize(filename) > 1000
        except Exception:
            return False


class InstallationEngine:
    """Основной движок установки компонентов для bat-контекста"""
    
    def __init__(self, system_manager: SystemManager, ui: UserInterface):
        self.system = system_manager
        self.ui = ui
        self.loc = system_manager.loc
        
        # Подавляем предупреждения для чистого вывода
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Проверка версии Python"""
        success, output = self.system.run_command([
            self.system.python_exe, "-c", 
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        ])
        
        if not success:
            return False, "Cannot detect Python version"
        
        try:
            python_version = float(output.strip())
            min_version = float(Config.MIN_PYTHON_VERSION)
            
            if python_version >= min_version:
                return True, f"Python {python_version} - OK"
            else:
                return False, f"Python {python_version} too old (need {Config.MIN_PYTHON_VERSION}+)"
        except ValueError:
            return False, f"Cannot parse Python version: {output}"
    
    def check_pytorch_version(self) -> Tuple[bool, str, Optional[str], str]:
        """Проверка версии PyTorch с определением статуса"""
        # Проверяем установлен ли PyTorch
        success, _ = self.system.run_command([
            self.system.python_exe, "-c", "import torch; print('installed')"
        ])
        
        if not success:
            return False, "PyTorch not found - installation required", None, 'missing'
        
        # Получаем версию
        success, version_output = self.system.run_command([
            self.system.python_exe, "-c", "import torch; print(torch.__version__)"
        ])
        
        if not success:
            return False, "Cannot detect PyTorch version", None, 'missing'
        
        current_version = version_output.strip()
        
        # Проверяем CUDA
        success, cuda_output = self.system.run_command([
            self.system.python_exe, "-c", "import torch; print(torch.cuda.is_available())"
        ])
        cuda_available = cuda_output.strip() == "True" if success else False
        
        # Определяем статус версии
        status = VersionManager.get_pytorch_status(current_version)
        
        # Формируем сообщение
        if status == 'compatible':
            message = f"PyTorch {current_version} - Compatible, CUDA: {cuda_available}"
            return True, message, current_version, status
        elif status == 'too_old':
            message = f"PyTorch {current_version} outdated (need {Config.TARGET_PYTORCH_VERSION}+), CUDA: {cuda_available}"
            return False, message, current_version, status
        else:  # too_new
            message = f"PyTorch {current_version} too new (recommended {Config.TARGET_PYTORCH_VERSION}-{Config.MAX_PYTORCH_VERSION}), CUDA: {cuda_available}"
            return False, message, current_version, status
    
    def install_pytorch(self, target_version: str = None) -> Tuple[bool, str]:
        """Установка или переустановка PyTorch"""
        if target_version is None:
            target_version = Config.TARGET_PYTORCH_VERSION
        
        print(f"Installing PyTorch {target_version} with CUDA 12.8...")
        print("This may take several minutes (~2.5GB download)")
        
        # Формируем команду установки
        cmd = [self.system.python_exe, "-m", "pip", "install"]
        cmd.extend(Config.get_pytorch_packages())
        cmd.extend(Config.get_pytorch_install_args())
        
        success, output = self.system.run_command(cmd, Config.PYTORCH_INSTALL_TIMEOUT)
        
        if success:
            # Проверяем результат установки
            check_success, version_output = self.system.run_command([
                self.system.python_exe, "-c", "import torch; print(torch.__version__)"
            ])
            
            if check_success:
                new_version = version_output.strip()
                return True, f"PyTorch {new_version} installed successfully"
            else:
                return True, "PyTorch installed (version check failed)"
        else:
            return False, f"PyTorch installation failed: {output[:300]}"
    
    def upgrade_pip(self) -> Tuple[bool, str]:
        """Обновление pip до последней версии"""
        success, output = self.system.run_command([
            self.system.python_exe, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        if success:
            return True, "pip upgraded successfully"
        else:
            return False, f"pip upgrade failed: {output[:200]}"
    
    def install_triton(self) -> Tuple[bool, str]:
        """Установка Triton Windows"""
        success, output = self.system.run_command([
            self.system.python_exe, "-m", "pip", "install", "-U", "triton-windows<3.4"
        ])
        
        if success:
            return True, "Triton installed"
        else:
            return False, f"Triton failed: {output[:200]}"
    
    def install_sageattention(self) -> Tuple[bool, str]:
        """Установка SageAttention из wheel файла"""
        # Скачивание wheel файла
        if not self.system.download_file(Config.SAGEATTENTION_WHEEL_URL, Config.SAGEATTENTION_WHEEL_FILE):
            return False, "Failed to download SageAttention wheel"
        
        # Установка из wheel
        success, output = self.system.run_command([
            self.system.python_exe, "-m", "pip", "install", Config.SAGEATTENTION_WHEEL_FILE
        ])
        
        # Очистка временного файла (bat-скрипт также удалит)
        try:
            os.remove(Config.SAGEATTENTION_WHEEL_FILE)
        except:
            pass
        
        if success:
            return True, "SageAttention installed"
        else:
            return False, f"SageAttention failed: {output[:200]}"
    
    def setup_include_libs(self) -> Tuple[bool, str]:
        """Скачивание и настройка папок include/libs"""
        # Проверяем существование папок
        if os.path.exists("include") and os.path.exists("libs"):
            return True, "include/libs folders already exist"
        
        # Скачивание архива
        if not self.system.download_file(Config.INCLUDE_LIBS_URL, Config.INCLUDE_LIBS_FILE):
            return False, "Failed to download include/libs archive"
        
        # Распаковка
        try:
            with zipfile.ZipFile(Config.INCLUDE_LIBS_FILE, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(Config.INCLUDE_LIBS_FILE)
            
            # Проверяем результат
            if os.path.exists("include") and os.path.exists("libs"):
                return True, "include/libs folders created"
            else:
                return False, "Extraction completed but folders not found"
                
        except Exception as e:
            return False, f"Extraction failed: {str(e)}"


class TRSAInstaller:
    """Главный класс приложения для bat-контекста"""
    
    def __init__(self):
        self.localization = LocalizationManager()
        self.ui = UserInterface(self.localization)
        self.system = SystemManager(self.localization)
        self.engine = InstallationEngine(self.system, self.ui)
    
    def run_installation(self) -> bool:
        """Запуск полного процесса установки"""
        try:
            # Выбор языка
            self.localization.ask_language_choice()
            
            # Заголовок
            self.ui.print_header()
            
            # Шаг 1: Проверка Python
            self.ui.print_step(1, 6, 'step_checking_python')
            success, message = self.engine.check_python_version()
            
            if success:
                self.ui.print_success(message)
            else:
                self.ui.print_error(message)
                return False
            
            # Шаг 2: Проверка и управление PyTorch
            self.ui.print_step(2, 6, 'step_checking_pytorch')
            pytorch_compatible, pytorch_message, current_version, status = self.engine.check_pytorch_version()
            
            if not pytorch_compatible:
                self.ui.print_warning(pytorch_message)
                
                if status == 'too_old':
                    if self.ui.ask_pytorch_action(current_version, 'update'):
                        print(f"\nInstalling PyTorch {Config.TARGET_PYTORCH_VERSION}...")
                        install_success, install_message = self.engine.install_pytorch()
                        
                        if install_success:
                            self.ui.print_success(install_message)
                        else:
                            self.ui.print_error(install_message)
                            self.ui.print_warning("Continuing with current PyTorch version...")
                
                elif status == 'too_new':
                    if self.ui.ask_pytorch_action(current_version, 'downgrade'):
                        print(f"\nDowngrading PyTorch to version {Config.TARGET_PYTORCH_VERSION}...")
                        downgrade_success, downgrade_message = self.engine.install_pytorch()
                        
                        if downgrade_success:
                            self.ui.print_success(downgrade_message)
                        else:
                            self.ui.print_error(downgrade_message)
                            self.ui.print_warning("Continuing with current PyTorch version...")
                
                elif status == 'missing':
                    print(f"\nInstalling PyTorch {Config.TARGET_PYTORCH_VERSION}...")
                    install_success, install_message = self.engine.install_pytorch()
                    
                    if install_success:
                        self.ui.print_success(install_message)
                    else:
                        self.ui.print_error(install_message)
                        return False
            else:
                self.ui.print_success(pytorch_message)
            
            # Шаги 3-6: Установка остальных компонентов
            remaining_steps = [
                (3, 'step_upgrading_pip', self.engine.upgrade_pip),
                (4, 'step_installing_triton', self.engine.install_triton),
                (5, 'step_installing_sage', self.engine.install_sageattention),
                (6, 'step_setting_up_libs', self.engine.setup_include_libs)
            ]
            
            all_success = True
            
            for step_num, step_key, method in remaining_steps:
                self.ui.print_step(step_num, 6, step_key)
                success, message = method()
                
                if success:
                    self.ui.print_success(message)
                else:
                    self.ui.print_error(message)
                    all_success = False
            
            # Финальный результат
            self.ui.print_final_result(all_success)
            return all_success
            
        except Exception as e:
            self.ui.print_error(f"Critical error: {str(e)}")
            return False


def main():
    """Точка входа в приложение для bat-контекста"""
    installer = None
    
    try:
        print("🔄 Initializing TRSA ComfyUI Accelerator in bat context...")
        print("🔄 Инициализация TRSA ComfyUI Accelerator в bat-контексте...")
        print()
        
        installer = TRSAInstaller()
        installer.run_installation()
        
    except KeyboardInterrupt:
        message = "Installation cancelled by user / Установка отменена пользователем"
        print(f"\n👋 {message}")
        
    except Exception as e:
        print(f"❌ Critical error / Критическая ошибка: {e}")
        
    finally:
        if installer:
            installer.ui.wait_for_exit()
        
        # В bat-контексте не делаем input(), bat-файл сам сделает pause
        print("\n🔄 Returning control to bat script...")
        print("🔄 Возврат управления bat-скрипту...")


if __name__ == "__main__":
    main()
