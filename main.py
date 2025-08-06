#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Accelerator v5.1 - Main Application Logic
Основная логика установки и управления зависимостями
"""
import subprocess
import urllib.request
import zipfile
import sys
import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Literal
from config import Config
from ui import LocalizationManager, UserInterface


class VersionManager:
    """Менеджер сравнения и управления версиями"""
    
    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """
        Интеллектуальное сравнение версий PyTorch
        
        Args:
            version1: первая версия для сравнения
            version2: вторая версия для сравнения
            
        Returns:
            -1 если version1 < version2
             0 если версии равны
             1 если version1 > version2
        """
        def clean_version(version: str) -> list:
            """Очистка версии от суффиксов и преобразование в список чисел"""
            try:
                clean_str = version.split('+')[0]  # Убираем +cu128 и подобные
                return [int(x) for x in clean_str.split('.')]
            except (ValueError, AttributeError):
                return [0]  # Fallback для некорректных версий
        
        try:
            v1_parts = clean_version(version1)
            v2_parts = clean_version(version2)
            
            # Выравниваем длину массивов нулями
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            # Сравниваем по частям
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0
            
        except Exception:
            # Fallback на строковое сравнение
            return -1 if version1 < version2 else (1 if version1 > version2 else 0)
    
    @classmethod
    def get_pytorch_status(cls, current_version: str) -> Literal['too_old', 'compatible', 'too_new']:
        """
        Определение статуса версии PyTorch
        
        Args:
            current_version: текущая версия PyTorch
            
        Returns:
            Статус совместимости версии
        """
        min_compare = cls.compare_versions(current_version, Config.TARGET_PYTORCH_VERSION)
        max_compare = cls.compare_versions(current_version, Config.MAX_PYTORCH_VERSION)
        
        if min_compare < 0:
            return 'too_old'
        elif max_compare > 0:
            return 'too_new'
        else:
            return 'compatible'


class SystemManager:
    """Менеджер системных операций и команд"""
    
    def __init__(self, localization: LocalizationManager):
        self.loc = localization
        self.python_exe = self._find_python_executable()
    
    def _find_python_executable(self) -> str:
        """
        Поиск исполняемого файла Python
        
        Returns:
            Путь к Python исполняемому файлу
            
        Raises:
            FileNotFoundError: если Python не найден
        """
        for exe in Config.get_python_executables():
            if os.path.exists(exe):
                return exe
        
        raise FileNotFoundError(self.loc.get_message('python_not_found'))
    
    def run_command(self, cmd: list, timeout: int = None) -> Tuple[bool, str]:
        """
        Выполнение системной команды с обработкой ошибок
        
        Args:
            cmd: команда для выполнения
            timeout: таймаут выполнения
            
        Returns:
            Tuple[успешность, вывод команды]
        """
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
            return False, self.loc.get_message('command_timeout')
        except Exception as e:
            return False, str(e)
    
    def download_file(self, url: str, filename: str) -> bool:
        """
        Безопасное скачивание файла
        
        Args:
            url: URL для скачивания
            filename: имя файла для сохранения
            
        Returns:
            True если скачивание успешно
        """
        try:
            urllib.request.urlretrieve(url, filename)
            return os.path.exists(filename) and os.path.getsize(filename) > 1000
        except Exception:
            return False


class InstallationEngine:
    """Основной движок установки компонентов"""
    
    def __init__(self, system_manager: SystemManager, ui: UserInterface):
        self.system = system_manager
        self.ui = ui
        self.loc = system_manager.loc
        
        # Подавляем предупреждения для чистого вывода
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    def check_python_version(self) -> Tuple[bool, str]:
        """
        Проверка версии Python
        
        Returns:
            Tuple[совместимость, сообщение о статусе]
        """
        success, output = self.system.run_command([
            self.system.python_exe, "-c", 
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        ])
        
        if not success:
            return False, self.loc.get_message('python_detect_error')
        
        try:
            python_version = float(output.strip())
            min_version = float(Config.MIN_PYTHON_VERSION)
            
            if python_version >= min_version:
                return True, f"Python {python_version} - OK"
            else:
                return False, self.loc.get_message(
                    'python_too_old',
                    version=python_version,
                    min_version=Config.MIN_PYTHON_VERSION
                )
        except ValueError:
            return False, self.loc.get_message('python_parse_error', output=output)
    
    def check_pytorch_version(self) -> Tuple[bool, str, Optional[str], str]:
        """
        Проверка версии PyTorch с определением статуса
        
        Returns:
            Tuple[совместимость, сообщение, текущая_версия, статус]
        """
        # Проверяем установлен ли PyTorch
        success, _ = self.system.run_command([
            self.system.python_exe, "-c", "import torch; print('installed')"
        ])
        
        if not success:
            return False, self.loc.get_message('pytorch_missing'), None, 'missing'
        
        # Получаем версию
        success, version_output = self.system.run_command([
            self.system.python_exe, "-c", "import torch; print(torch.__version__)"
        ])
        
        if not success:
            return False, self.loc.get_message('pytorch_detect_error'), None, 'missing'
        
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
        """
        Установка или переустановка PyTorch
        
        Args:
            target_version: целевая версия (по умолчанию из конфига)
            
        Returns:
            Tuple[успешность, сообщение о результате]
        """
        if target_version is None:
            target_version = Config.TARGET_PYTORCH_VERSION
        
        print(self.loc.get_message('installing_pytorch_cuda', version=target_version))
        print(self.loc.get_message('download_warning'))
        
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
                return True, self.loc.get_message('pytorch_installed_success', version=new_version)
            else:
                return True, self.loc.get_message('pytorch_installed_no_check')
        else:
            return False, self.loc.get_message('pytorch_install_failed', error=output[:300])
    
    def upgrade_pip(self) -> Tuple[bool, str]:
        """Обновление pip до последней версии"""
        success, output = self.system.run_command([
            self.system.python_exe, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        if success:
            return True, self.loc.get_message('pip_upgraded')
        else:
            return False, self.loc.get_message('pip_upgrade_failed', error=output[:200])
    
    def install_triton(self) -> Tuple[bool, str]:
        """Установка Triton Windows"""
        success, output = self.system.run_command([
            self.system.python_exe, "-m", "pip", "install", "-U", "triton-windows<3.4"
        ])
        
        if success:
            return True, self.loc.get_message('triton_installed')
        else:
            return False, self.loc.get_message('triton_failed', error=output[:200])
    
    def install_sageattention(self) -> Tuple[bool, str]:
        """Установка SageAttention из wheel файла"""
        # Скачивание wheel файла
        if not self.system.download_file(Config.SAGEATTENTION_WHEEL_URL, Config.SAGEATTENTION_WHEEL_FILE):
            return False, self.loc.get_message('sage_download_failed')
        
        # Установка из wheel
        success, output = self.system.run_command([
            self.system.python_exe, "-m", "pip", "install", Config.SAGEATTENTION_WHEEL_FILE
        ])
        
        # Очистка временного файла
        try:
            os.remove(Config.SAGEATTENTION_WHEEL_FILE)
        except:
            pass
        
        if success:
            return True, self.loc.get_message('sage_installed')
        else:
            return False, self.loc.get_message('sage_failed', error=output[:200])
    
    def setup_include_libs(self) -> Tuple[bool, str]:
        """Скачивание и настройка папок include/libs"""
        # Проверяем существование папок
        if os.path.exists("include") and os.path.exists("libs"):
            return True, self.loc.get_message('libs_exist')
        
        # Скачивание архива
        if not self.system.download_file(Config.INCLUDE_LIBS_URL, Config.INCLUDE_LIBS_FILE):
            return False, self.loc.get_message('libs_download_failed')
        
        # Распаковка
        try:
            with zipfile.ZipFile(Config.INCLUDE_LIBS_FILE, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(Config.INCLUDE_LIBS_FILE)
            
            # Проверяем результат
            if os.path.exists("include") and os.path.exists("libs"):
                return True, self.loc.get_message('libs_created')
            else:
                return False, self.loc.get_message('libs_extract_no_folders')
                
        except Exception as e:
            return False, self.loc.get_message('libs_extract_failed', error=str(e))


class TRSAInstaller:
    """Главный класс приложения TRSA ComfyUI Accelerator"""
    
    def __init__(self):
        self.localization = LocalizationManager()
        self.ui = UserInterface(self.localization)
        self.system = SystemManager(self.localization)
        self.engine = InstallationEngine(self.system, self.ui)
    
    def run_installation(self) -> bool:
        """
        Запуск полного процесса установки
        
        Returns:
            True если установка прошла успешно
        """
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
                        print(f"\n{self.localization.get_message('installing_pytorch', version=Config.TARGET_PYTORCH_VERSION)}...")
                        install_success, install_message = self.engine.install_pytorch()
                        
                        if install_success:
                            self.ui.print_success(install_message)
                        else:
                            self.ui.print_error(install_message)
                            self.ui.print_warning(self.localization.get_message('continuing_current_pytorch'))
                
                elif status == 'too_new':
                    if self.ui.ask_pytorch_action(current_version, 'downgrade'):
                        print(f"\n{self.localization.get_message('downgrading_pytorch', version=Config.TARGET_PYTORCH_VERSION)}...")
                        downgrade_success, downgrade_message = self.engine.install_pytorch()
                        
                        if downgrade_success:
                            self.ui.print_success(downgrade_message)
                        else:
                            self.ui.print_error(downgrade_message)
                            self.ui.print_warning(self.localization.get_message('continuing_current_pytorch'))
                
                elif status == 'missing':
                    print(f"\n{self.localization.get_message('installing_pytorch', version=Config.TARGET_PYTORCH_VERSION)}...")
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
            self.ui.print_error(self.localization.get_message('critical_error', error=str(e)))
            return False


def main():
    """Точка входа в приложение"""
    installer = None
    
    try:
        installer = TRSAInstaller()
        installer.run_installation()
        
    except KeyboardInterrupt:
        message = installer.localization.get_message('installation_cancelled') if installer else 'Installation cancelled by user'
        print(f"\n👋 {message}")
        
    except Exception as e:
        error_message = installer.localization.get_message('critical_error', error=str(e)) if installer else f"Critical error: {e}"
        print(f"❌ {error_message}")
        
    finally:
        if installer:
            installer.ui.wait_for_exit()
        else:
            print("\nPress Enter to exit... / Нажмите Enter для выхода...")
            input()


if __name__ == "__main__":
    main()
