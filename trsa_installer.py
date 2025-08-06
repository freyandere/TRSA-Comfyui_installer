#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Accelerator v5.1 - Consolidated Single-File Implementation
Консолидированная реализация в одном файле без потери функционала
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import locale
import warnings
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Консолидированная конфигурация приложения"""
    APP_VERSION: str = "5.1"
    MIN_PYTHON_VERSION: str = "3.12"
    TARGET_PYTORCH_VERSION: str = "2.7.1"
    MAX_PYTORCH_VERSION: str = "2.7.9"
    
    # URLs
    PYTORCH_INSTALL_URL: str = "https://download.pytorch.org/whl/cu128"
    SAGEATTENTION_WHEEL_URL: str = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
    INCLUDE_LIBS_URL: str = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/python_3.12.7_include_libs.zip"
    
    # Таймауты
    COMMAND_TIMEOUT: int = 300
    PYTORCH_INSTALL_TIMEOUT: int = 900
    
    @property
    def python_executables(self) -> List[str]:
        return ["./python.exe", "python.exe", "python"]
    
    @property
    def pytorch_packages(self) -> List[str]:
        return [f"torch=={self.TARGET_PYTORCH_VERSION}", "torchvision", "torchaudio"]


class LocalizedMessages:
    """Консолидированная система сообщений с параметризацией"""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Оптимизированные шаблоны сообщений без дублирования"""
        return {
            'en': {
                # Core messages
                'app_title': f'TRSA ComfyUI Accelerator v{AppConfig.APP_VERSION} - Smart Installation',
                'step_format': '{step}/{total} {action}...',
                'version_status': '{component} {version} - {status}',
                'user_choice': '1-{action}, 2-Continue with current version: ',
                
                # Status templates
                'pytorch_status': {
                    'compatible': 'Compatible, CUDA: {cuda}',
                    'too_old': 'outdated (need {target}+), CUDA: {cuda}',
                    'too_new': 'too new (recommended {target}-{max}), CUDA: {cuda}',
                    'missing': 'not found - installation required'
                },
                
                # Actions
                'actions': {
                    'checking_python': 'Checking Python version',
                    'checking_pytorch': 'Checking PyTorch version',
                    'installing_pytorch': 'Installing PyTorch {version}',
                    'upgrading_pip': 'Upgrading pip',
                    'installing_triton': 'Installing Triton',
                    'installing_sage': 'Installing SageAttention'
                }
            },
            'ru': {
                # Core messages
                'app_title': f'TRSA ComfyUI Accelerator v{AppConfig.APP_VERSION} - Умная установка',
                'step_format': '{step}/{total} {action}...',
                'version_status': '{component} {version} - {status}',
                'user_choice': '1-{action}, 2-Продолжить с текущей версией: ',
                
                # Status templates
                'pytorch_status': {
                    'compatible': 'Совместима, CUDA: {cuda}',
                    'too_old': 'устарела (нужна {target}+), CUDA: {cuda}',
                    'too_new': 'слишком новая (рекомендуется {target}-{max}), CUDA: {cuda}',
                    'missing': 'не найдена - требуется установка'
                },
                
                # Actions
                'actions': {
                    'checking_python': 'Проверка версии Python',
                    'checking_pytorch': 'Проверка версии PyTorch',
                    'installing_pytorch': 'Установка PyTorch {version}',
                    'upgrading_pip': 'Обновление pip',
                    'installing_triton': 'Установка Triton',
                    'installing_sage': 'Установка SageAttention'
                }
            }
        }
    
    def get(self, key: str, **kwargs) -> str:
        """Получить сообщение с форматированием"""
        # Поддержка вложенных ключей (например, 'pytorch_status.compatible')
        keys = key.split('.')
        message = self._templates[self.language]
        
        for k in keys:
            message = message.get(k, key)
            if isinstance(message, str):
                break
        
        if isinstance(message, str) and kwargs:
            try:
                return message.format(**kwargs)
            except (KeyError, ValueError):
                return message
        
        return str(message)


class TRSAInstaller:
    """Консолидированный установщик с минимальным API"""
    
    def __init__(self):
        self.config = AppConfig()
        self.language = self._detect_language()
        self.messages = LocalizedMessages(self.language)
        self.python_exe = self._find_python_executable()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    def _detect_language(self) -> str:
        """Упрощенное определение языка"""
        # Проверка переменных окружения
        for var in ['LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE']:
            if os.environ.get(var, '').lower().startswith('ru'):
                return 'ru'
        
        # Windows Registry (быстрая проверка)
        if os.name == 'nt':
            try:
                result = subprocess.run([
                    'reg', 'query', 'HKCU\\Control Panel\\International',
                    '/v', 'LocaleName'
                ], capture_output=True, text=True, timeout=3)
                if 'ru' in result.stdout.lower():
                    return 'ru'
            except:
                pass
        
        return 'en'
    
    def _find_python_executable(self) -> str:
        """Поиск Python исполняемого файла"""
        for exe in self.config.python_executables:
            if os.path.exists(exe):
                return exe
        raise FileNotFoundError("Python executable not found")
    
    def _run_command(self, cmd: List[str], timeout: int = None) -> Tuple[bool, str]:
        """Выполнение команды с обработкой ошибок"""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout or self.config.COMMAND_TIMEOUT
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timeout"
        except Exception as e:
            return False, str(e)
    
    def _print_step(self, step: int, total: int, action_key: str, **kwargs):
        """Вывод шага установки"""
        action = self.messages.get(f'actions.{action_key}', **kwargs)
        message = self.messages.get('step_format', step=step, total=total, action=action)
        print(f"\n{message}")
    
    def _ask_user_choice(self, action: str) -> bool:
        """Запрос выбора пользователя"""
        choice_msg = self.messages.get('user_choice', action=action)
        while True:
            try:
                choice = input(choice_msg).strip()
                if choice in ['1', '2']:
                    return choice == '1'
                print("❌ Please enter 1 or 2")
            except (EOFError, KeyboardInterrupt):
                return False
    
    def check_pytorch_version(self) -> Tuple[bool, str, Optional[str], str]:
        """Консолидированная проверка PyTorch"""
        # Проверка установки
        success, _ = self._run_command([self.python_exe, "-c", "import torch"])
        if not success:
            status_msg = self.messages.get('pytorch_status.missing')
            return False, f"PyTorch {status_msg}", None, 'missing'
        
        # Получение версии
        success, version_output = self._run_command([
            self.python_exe, "-c", "import torch; print(torch.__version__)"
        ])
        if not success:
            return False, "Cannot detect PyTorch version", None, 'missing'
        
        current_version = version_output.strip()
        
        # Проверка CUDA
        success, cuda_output = self._run_command([
            self.python_exe, "-c", "import torch; print(torch.cuda.is_available())"
        ])
        cuda_available = cuda_output.strip() == "True" if success else False
        
        # Определение статуса
        status = self._get_pytorch_status(current_version)
        status_msg = self.messages.get(
            f'pytorch_status.{status}',
            cuda=cuda_available,
            target=self.config.TARGET_PYTORCH_VERSION,
            max=self.config.MAX_PYTORCH_VERSION
        )
        
        version_msg = self.messages.get(
            'version_status',
            component='PyTorch',
            version=current_version,
            status=status_msg
        )
        
        return status == 'compatible', version_msg, current_version, status
    
    def _get_pytorch_status(self, version: str) -> Literal['too_old', 'compatible', 'too_new']:
        """Определение статуса версии PyTorch"""
        def version_tuple(v: str) -> tuple:
            try:
                return tuple(int(x) for x in v.split('+')[0].split('.'))
            except:
                return (0,)
        
        current = version_tuple(version)
        target = version_tuple(self.config.TARGET_PYTORCH_VERSION)
        max_ver = version_tuple(self.config.MAX_PYTORCH_VERSION)
        
        if current < target:
            return 'too_old'
        elif current > max_ver:
            return 'too_new'
        else:
            return 'compatible'
    
    def install_pytorch(self) -> Tuple[bool, str]:
        """Установка PyTorch"""
        print(self.messages.get('actions.installing_pytorch', version=self.config.TARGET_PYTORCH_VERSION))
        print("This may take several minutes (~2.5GB download)")
        
        cmd = [self.python_exe, "-m", "pip", "install", "--force-reinstall", 
               "--index-url", self.config.PYTORCH_INSTALL_URL] + self.config.pytorch_packages
        
        success, output = self._run_command(cmd, self.config.PYTORCH_INSTALL_TIMEOUT)
        
        if success:
            return True, f"PyTorch {self.config.TARGET_PYTORCH_VERSION} installed successfully"
        else:
            return False, f"PyTorch installation failed: {output[:300]}"
    
    def run(self) -> bool:
        """Основной процесс установки"""
        try:
            # Выбор языка
            if not sys.stdin.isatty():  # Автоматическое определение в bat-контексте
                pass
            else:
                choice = input("Choose language / Выберите язык: 1-English, 2-Русский: ")
                if choice == '2':
                    self.language = 'ru'
                    self.messages = LocalizedMessages('ru')
            
            # Заголовок
            print(self.messages.get('app_title'))
            print('=' * 60)
            
            # Установка
            steps = [
                ('checking_python', self._check_python),
                ('checking_pytorch', self._handle_pytorch),
                ('upgrading_pip', self._upgrade_pip),
                ('installing_triton', self._install_triton),
                ('installing_sage', self._install_sageattention)
            ]
            
            all_success = True
            for i, (action_key, method) in enumerate(steps, 1):
                self._print_step(i, len(steps), action_key)
                success = method()
                if not success:
                    all_success = False
            
            print(f"\n{'✅ Installation completed!' if all_success else '⚠️ Some components failed'}")
            return all_success
            
        except KeyboardInterrupt:
            print("\n❌ Installation cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Critical error: {e}")
            return False
    
    def _check_python(self) -> bool:
        """Проверка версии Python"""
        success, output = self._run_command([
            self.python_exe, "-c", 
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        ])
        
        if success and float(output.strip()) >= float(self.config.MIN_PYTHON_VERSION):
            print(f"✅ Python {output.strip()} - OK")
            return True
        else:
            print(f"❌ Python version check failed")
            return False
    
    def _handle_pytorch(self) -> bool:
        """Обработка PyTorch с интерактивностью"""
        compatible, message, current_version, status = self.check_pytorch_version()
        
        if compatible:
            print(f"✅ {message}")
            return True
        
        print(f"⚠️ {message}")
        
        if status == 'missing':
            return self.install_pytorch()[0]
        
        action = 'Update' if status == 'too_old' else 'Downgrade'
        if self._ask_user_choice(action):
            return self.install_pytorch()[0]
        else:
            print("Continuing with current version...")
            return True
    
    def _upgrade_pip(self) -> bool:
        """Обновление pip"""
        success, _ = self._run_command([self.python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        print(f"{'✅' if success else '❌'} pip {'upgraded' if success else 'upgrade failed'}")
        return success
    
    def _install_triton(self) -> bool:
        """Установка Triton"""
        success, _ = self._run_command([self.python_exe, "-m", "pip", "install", "-U", "triton-windows<3.4"])
        print(f"{'✅' if success else '❌'} Triton {'installed' if success else 'failed'}")
        return success
    
    def _install_sageattention(self) -> bool:
        """Установка SageAttention"""
        wheel_file = "sageattention.whl"
        try:
            urllib.request.urlretrieve(self.config.SAGEATTENTION_WHEEL_URL, wheel_file)
            success, _ = self._run_command([self.python_exe, "-m", "pip", "install", wheel_file])
            os.remove(wheel_file)
            print(f"{'✅' if success else '❌'} SageAttention {'installed' if success else 'failed'}")
            return success
        except:
            print("❌ SageAttention download failed")
            return False


def main():
    """Точка входа"""
    installer = TRSAInstaller()
    installer.run()


if __name__ == "__main__":
    main()
