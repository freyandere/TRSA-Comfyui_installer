#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Accelerator v5.0 - With PyTorch Version Management
Интегрированная проверка и управление версиями PyTorch
"""
import subprocess
import urllib.request
import zipfile
import sys
import os
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional
from packaging import version


class EnhancedSimpleInstaller:
    """
    Простой установщик ComfyUI Accelerator с интеллектуальным управлением PyTorch
    """
    
    # Константы для версий
    MIN_PYTHON_VERSION = "3.12"
    MIN_PYTORCH_VERSION = "2.7.1"
    PYTORCH_INSTALL_URL = "https://download.pytorch.org/whl/cu128"
    
    def __init__(self):
        self.python_exe = self._find_python()
        self.language = self._detect_language_modern()
        self.messages = self._load_messages()
        
        # Подавляем DeprecationWarning для чистого вывода
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
    def _find_python(self) -> str:
        """Поиск исполняемого файла Python"""
        for exe in ["python.exe", "python"]:
            if os.path.exists(exe):
                return exe
        raise FileNotFoundError("Python executable not found")
    
    def _detect_language_modern(self) -> str:
        """Современный способ определения языка (без DeprecationWarning)"""
        try:
            # Переменные окружения
            for env_var in ['LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE']:
                lang = os.environ.get(env_var, '').lower()
                if lang.startswith('ru'):
                    return 'ru'
            
            # Windows - проверяем реестр
            if os.name == 'nt':
                try:
                    result = subprocess.run([
                        'reg', 'query', 'HKCU\\Control Panel\\International',
                        '/v', 'LocaleName'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'LocaleName' in line and 'ru' in line.lower():
                                return 'ru'
                except:
                    pass
            
            # Современный locale API
            try:
                import locale
                current_locale = locale.getlocale()
                if current_locale[0] and current_locale[0].lower().startswith('ru'):
                    return 'ru'
                    
                encoding = locale.getencoding()
                if 'ru' in encoding.lower():
                    return 'ru'
            except:
                pass
                
        except Exception:
            pass
        
        return 'en'
    
    def _load_messages(self) -> Dict[str, Dict[str, str]]:
        """Загрузка сообщений для двух языков"""
        return {
            'ru': {
                'header': '🚀 ComfyUI Accelerator v5.0 - Умная установка',
                'lang_choice': 'Выберите язык / Choose language: 1-English, 2-Русский: ',
                'checking_python': 'Проверка версии Python',
                'checking_pytorch': 'Проверка версии PyTorch',
                'pytorch_ok': 'PyTorch версии актуальной',
                'pytorch_missing': 'PyTorch не найден - требуется установка',
                'pytorch_outdated': 'PyTorch устарел - рекомендуется обновление',
                'pytorch_update_prompt': 'Обновить PyTorch до версии 2.7.1?',
                'pytorch_warning': 'ВНИМАНИЕ: Старая версия может вызвать ошибки!',
                'pytorch_installing': 'Установка PyTorch 2.7.1',
                'pytorch_choice': '1-Обновить PyTorch, 2-Продолжить с текущей версией: ',
                'upgrading_pip': 'Обновление pip',
                'installing_triton': 'Установка Triton',
                'installing_sage': 'Установка SageAttention',
                'setting_up_libs': 'Настройка include/libs',
                'success': 'Установка завершена успешно!',
                'restart_note': '💡 Перезапустите ComfyUI для применения ускорения',
                'some_failed': 'Некоторые компоненты не установились',
                'press_enter': 'Нажмите Enter для выхода...'
            },
            'en': {
                'header': '🚀 ComfyUI Accelerator v5.0 - Smart Installation',
                'lang_choice': 'Choose language / Выберите язык: 1-English, 2-Русский: ',
                'checking_python': 'Checking Python version',
                'checking_pytorch': 'Checking PyTorch version',
                'pytorch_ok': 'PyTorch version is up-to-date',
                'pytorch_missing': 'PyTorch not found - installation required',
                'pytorch_outdated': 'PyTorch is outdated - update recommended',
                'pytorch_update_prompt': 'Update PyTorch to version 2.7.1?',
                'pytorch_warning': 'WARNING: Old version may cause errors!',
                'pytorch_installing': 'Installing PyTorch 2.7.1',
                'pytorch_choice': '1-Update PyTorch, 2-Continue with current version: ',
                'upgrading_pip': 'Upgrading pip',
                'installing_triton': 'Installing Triton',
                'installing_sage': 'Installing SageAttention',
                'setting_up_libs': 'Setting up include/libs',
                'success': 'Installation completed successfully!',
                'restart_note': '💡 Restart ComfyUI to apply 2-3x speed improvements',
                'some_failed': 'Some components failed to install',
                'press_enter': 'Press Enter to exit...'
            }
        }
    
    def msg(self, key: str) -> str:
        """Получить сообщение на текущем языке"""
        return self.messages[self.language].get(key, key)
    
    def _run_command(self, cmd: list, timeout: int = 300) -> Tuple[bool, str]:
        """Выполнение команды с обработкой ошибок"""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8'
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timeout"
        except Exception as e:
            return False, str(e)
    
    def _download_file(self, url: str, filename: str) -> bool:
        """Скачивание файла с обработкой ошибок"""
        try:
            urllib.request.urlretrieve(url, filename)
            return os.path.exists(filename) and os.path.getsize(filename) > 1000
        except Exception:
            return False
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Проверка версии Python (требуется 3.12+)"""
        success, output = self._run_command([
            self.python_exe, "-c", 
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        ])
        
        if success:
            try:
                python_version = float(output.strip())
                min_version = float(self.MIN_PYTHON_VERSION)
                if python_version >= min_version:
                    return True, f"Python {python_version} - OK"
                return False, f"Python {python_version} too old (need {self.MIN_PYTHON_VERSION}+)"
            except ValueError:
                return False, f"Cannot parse Python version: {output}"
        return False, "Cannot detect Python version"
    
    def check_pytorch_version(self) -> Tuple[bool, str, Optional[str]]:
        """
        Проверка версии PyTorch
        Returns: (pytorch_installed, message, current_version)
        """
        # Проверяем наличие PyTorch
        success, output = self._run_command([
            self.python_exe, "-c", "import torch; print('installed')"
        ])
        
        if not success:
            return False, self.msg('pytorch_missing'), None
        
        # Получаем версию PyTorch
        success, version_output = self._run_command([
            self.python_exe, "-c", "import torch; print(torch.__version__)"
        ])
        
        if not success:
            return False, "Cannot detect PyTorch version", None
        
        current_version = version_output.strip()
        
        # Проверяем CUDA
        success, cuda_output = self._run_command([
            self.python_exe, "-c", "import torch; print(torch.cuda.is_available())"
        ])
        cuda_available = cuda_output.strip() == "True" if success else False
        
        # Сравниваем версии
        try:
            # Пытаемся использовать packaging для сравнения версий
            success, _ = self._run_command([
                self.python_exe, "-c", 
                f"from packaging import version; "
                f"exit(0 if version.parse('{current_version}') >= version.parse('{self.MIN_PYTORCH_VERSION}') else 1)"
            ])
            
            if success:
                status_msg = f"PyTorch {current_version} - OK, CUDA: {cuda_available}"
                return True, status_msg, current_version
            else:
                status_msg = f"PyTorch {current_version} outdated (need {self.MIN_PYTORCH_VERSION}+), CUDA: {cuda_available}"
                return False, status_msg, current_version
                
        except Exception:
            # Fallback: простое сравнение версий
            try:
                current_parts = current_version.split('.')
                min_parts = self.MIN_PYTORCH_VERSION.split('.')
                
                for i in range(min(len(current_parts), len(min_parts))):
                    current_part = int(current_parts[i].split('+')[0])  # Убираем +cu128 если есть
                    min_part = int(min_parts[i])
                    
                    if current_part > min_part:
                        return True, f"PyTorch {current_version} - OK", current_version
                    elif current_part < min_part:
                        return False, f"PyTorch {current_version} outdated", current_version
                
                return True, f"PyTorch {current_version} - OK", current_version
                
            except Exception:
                return False, f"PyTorch {current_version} - version check failed", current_version
    
    def ask_pytorch_update(self, current_version: str) -> bool:
        """Запрос пользователя об обновлении PyTorch"""
        print(f"\n⚠️ {self.msg('pytorch_warning')}")
        print(f"📊 Current version: {current_version}")
        print(f"📊 Required version: {self.MIN_PYTORCH_VERSION}+")
        print(f"💡 SageAttention2 requires PyTorch {self.MIN_PYTORCH_VERSION}+ for optimal performance")
        print()
        
        while True:
            choice = input(self.msg('pytorch_choice')).strip()
            
            if choice == '1':
                return True
            elif choice == '2':
                print(f"\n⚠️ Continuing with PyTorch {current_version}")
                print("❗ Warning: You may encounter compatibility issues!")
                print("💡 Consider updating PyTorch later if problems occur")
                return False
            else:
                print("❌ Please enter 1 or 2")
    
    def install_pytorch(self) -> Tuple[bool, str]:
        """Установка PyTorch 2.7.1 с CUDA 12.8"""
        print(f"📦 Installing PyTorch {self.MIN_PYTORCH_VERSION} with CUDA 12.8...")
        print("⏳ This may take several minutes (~2.5GB download)")
        
        success, output = self._run_command([
            self.python_exe, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--force-reinstall",
            "--index-url", self.PYTORCH_INSTALL_URL
        ], timeout=900)  # 15 минут таймаут для больших загрузок
        
        if success:
            # Проверяем установленную версию
            check_success, version_output = self._run_command([
                self.python_exe, "-c", "import torch; print(torch.__version__)"
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
        success, output = self._run_command([
            self.python_exe, "-m", "pip", "install", "--upgrade", "pip"
        ])
        return success, "pip upgraded successfully" if success else f"pip upgrade failed: {output[:200]}"
    
    def install_triton(self) -> Tuple[bool, str]:
        """Установка Triton Windows"""
        success, output = self._run_command([
            self.python_exe, "-m", "pip", "install", "-U", "triton-windows<3.4"
        ])
        return success, "Triton installed" if success else f"Triton failed: {output[:200]}"
    
    def install_sageattention(self) -> Tuple[bool, str]:
        """Установка SageAttention"""
        wheel_url = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
        wheel_file = "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
        
        # Скачивание wheel
        if not self._download_file(wheel_url, wheel_file):
            return False, "Failed to download SageAttention wheel"
        
        # Установка из wheel
        success, output = self._run_command([
            self.python_exe, "-m", "pip", "install", wheel_file
        ])
        
        # Очистка
        try:
            os.remove(wheel_file)
        except:
            pass
            
        return success, "SageAttention installed" if success else f"SageAttention failed: {output[:200]}"
    
    def setup_include_libs(self) -> Tuple[bool, str]:
        """Скачивание и распаковка папок include/libs"""
        if os.path.exists("include") and os.path.exists("libs"):
            return True, "include/libs folders already exist"
        
        zip_url = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/python_3.12.7_include_libs.zip"
        zip_file = "include_libs.zip"
        
        # Скачивание
        if not self._download_file(zip_url, zip_file):
            return False, "Failed to download include/libs archive"
        
        # Распаковка
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zip_file)
            
            if os.path.exists("include") and os.path.exists("libs"):
                return True, "include/libs folders created"
            return False, "Extraction completed but folders not found"
            
        except Exception as e:
            return False, f"Extraction failed: {str(e)}"
    
    def choose_language(self):
        """Выбор языка пользователем"""
        print(self.msg('lang_choice'), end='')
        try:
            choice = input().strip()
            if choice == '1':
                self.language = 'en'
            elif choice == '2':
                self.language = 'ru'
        except:
            pass  # Оставляем автоопределенный язык
    
    def run_installation(self) -> bool:
        """Выполнение полной установки с проверкой PyTorch"""
        # Выбор языка
        self.choose_language()
        
        print(self.msg('header'))
        print("=" * 60)
        
        # Шаг 1: Проверка Python
        print(f"\n1/6 {self.msg('checking_python')}...")
        success, message = self.check_python_version()
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            return False
        
        # Шаг 2: Проверка PyTorch
        print(f"\n2/6 {self.msg('checking_pytorch')}...")
        pytorch_ok, pytorch_message, current_pytorch = self.check_pytorch_version()
        
        if not pytorch_ok:
            print(f"⚠️ {pytorch_message}")
            
            if current_pytorch:
                # PyTorch установлен, но устарел
                if self.ask_pytorch_update(current_pytorch):
                    print(f"\n🔄 {self.msg('pytorch_installing')}...")
                    install_success, install_message = self.install_pytorch()
                    if install_success:
                        print(f"✅ {install_message}")
                    else:
                        print(f"❌ {install_message}")
                        print("⚠️ Continuing with current PyTorch version...")
            else:
                # PyTorch не установлен вообще
                print(f"\n🔄 {self.msg('pytorch_installing')}...")
                install_success, install_message = self.install_pytorch()
                if install_success:
                    print(f"✅ {install_message}")
                else:
                    print(f"❌ {install_message}")
                    return False
        else:
            print(f"✅ {pytorch_message}")
        
        # Остальные шаги установки
        remaining_steps = [
            (f"3/6 {self.msg('upgrading_pip')}", self.upgrade_pip),
            (f"4/6 {self.msg('installing_triton')}", self.install_triton),
            (f"5/6 {self.msg('installing_sage')}", self.install_sageattention),
            (f"6/6 {self.msg('setting_up_libs')}", self.setup_include_libs)
        ]
        
        all_success = True
        for description, method in remaining_steps:
            print(f"\n{description}...")
            success, message = method()
            
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                all_success = False
        
        print("\n" + "=" * 60)
        if all_success:
            print(f"🎉 {self.msg('success')}")
            print(self.msg('restart_note'))
        else:
            print(f"⚠️ {self.msg('some_failed')}")
        
        return all_success


if __name__ == "__main__":
    try:
        installer = EnhancedSimpleInstaller()
        installer.run_installation()
    except KeyboardInterrupt:
        print("\n👋 Installation cancelled by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        print()
        try:
            print(installer.msg('press_enter'))
        except:
            print("Press Enter to exit... / Нажмите Enter для выхода...")
        input()
