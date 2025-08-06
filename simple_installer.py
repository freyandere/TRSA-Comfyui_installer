#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Accelerator v5.0 - Полностью локализованная версия
"""
import subprocess
import urllib.request
import zipfile
import sys
import os
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional


class EnhancedSimpleInstaller:
    """
    TRSA ComfyUI Accelerator с полной двуязычной поддержкой
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
        """Загрузка ВСЕХ сообщений для двух языков"""
        return {
            'ru': {
                # Основные сообщения интерфейса
                'header': 'TRSA ComfyUI Accelerator v5.0 - Умная установка',
                'lang_choice': 'Выберите язык / Choose language: 1-English, 2-Русский: ',
                'checking_python': 'Проверка версии Python',
                'checking_pytorch': 'Проверка версии PyTorch',
                'pytorch_ok': 'PyTorch версия актуальна',
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
                'restart_note': 'Перезапустите ComfyUI для применения ускорения',
                'some_failed': 'Некоторые компоненты не установились',
                'press_enter': 'Нажмите Enter для выхода...',
                
                # Дополнительные сообщения для полной локализации
                'current_version': 'Текущая версия',
                'required_version': 'Требуемая версия',
                'sage_requirement': 'SageAttention2 требует PyTorch {version}+ для оптимальной производительности',
                'continuing_with_version': 'Продолжаем с PyTorch {version}',
                'compatibility_warning': 'Предупреждение: Могут возникнуть проблемы совместимости!',
                'consider_updating': 'Рассмотрите возможность обновления PyTorch позже при возникновении проблем',
                'invalid_choice': 'Пожалуйста, введите 1 или 2',
                'installing_pytorch_cuda': 'Установка PyTorch {version} с CUDA 12.8...',
                'download_time_warning': 'Это может занять несколько минут (~2.5GB загрузка)',
                'pytorch_installed_success': 'PyTorch {version} установлен успешно',
                'pytorch_installed_no_check': 'PyTorch установлен (проверка версии не удалась)',
                'pytorch_install_failed': 'Установка PyTorch не удалась: {error}',
                'continuing_current_pytorch': 'Продолжаем с текущей версией PyTorch...',
                
                # Сообщения об ошибках версий
                'python_too_old': 'Python {version} слишком старый (требуется {min_version}+)',
                'python_version_parse_error': 'Не удается распознать версию Python: {output}',
                'python_version_detect_error': 'Не удается определить версию Python',
                'pytorch_version_detect_error': 'Не удается определить версию PyTorch',
                'pytorch_version_check_failed': 'PyTorch {version} - проверка версии не удалась',
                
                # Сообщения об установке компонентов
                'pip_upgraded': 'pip обновлен успешно',
                'pip_upgrade_failed': 'Обновление pip не удалось: {error}',
                'triton_installed': 'Triton установлен',
                'triton_failed': 'Установка Triton не удалась: {error}',
                'sage_download_failed': 'Не удалось скачать wheel файл SageAttention',
                'sage_installed': 'SageAttention установлен',
                'sage_failed': 'Установка SageAttention не удалась: {error}',
                
                # Сообщения об include/libs
                'include_libs_exist': 'Папки include/libs уже существуют',
                'include_libs_download_failed': 'Не удалось скачать архив include/libs',
                'include_libs_created': 'Папки include/libs созданы',
                'include_libs_extract_no_folders': 'Распаковка завершена, но папки не найдены',
                'include_libs_extract_failed': 'Распаковка не удалась: {error}',
                
                # Системные сообщения
                'installation_cancelled': 'Установка отменена пользователем',
                'critical_error': 'Критическая ошибка: {error}',
                'command_timeout': 'Превышено время ожидания команды',
                'python_not_found': 'Исполняемый файл Python не найден'
            },
            'en': {
                # Основные сообщения интерфейса
                'header': 'TRSA ComfyUI Accelerator v5.0 - Smart Installation',
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
                'restart_note': 'Restart ComfyUI to apply 2-3x speed improvements',
                'some_failed': 'Some components failed to install',
                'press_enter': 'Press Enter to exit...',
                
                # Дополнительные сообщения для полной локализации
                'current_version': 'Current version',
                'required_version': 'Required version',
                'sage_requirement': 'SageAttention2 requires PyTorch {version}+ for optimal performance',
                'continuing_with_version': 'Continuing with PyTorch {version}',
                'compatibility_warning': 'Warning: You may encounter compatibility issues!',
                'consider_updating': 'Consider updating PyTorch later if problems occur',
                'invalid_choice': 'Please enter 1 or 2',
                'installing_pytorch_cuda': 'Installing PyTorch {version} with CUDA 12.8...',
                'download_time_warning': 'This may take several minutes (~2.5GB download)',
                'pytorch_installed_success': 'PyTorch {version} installed successfully',
                'pytorch_installed_no_check': 'PyTorch installed (version check failed)',
                'pytorch_install_failed': 'PyTorch installation failed: {error}',
                'continuing_current_pytorch': 'Continuing with current PyTorch version...',
                
                # Сообщения об ошибках версий
                'python_too_old': 'Python {version} too old (need {min_version}+)',
                'python_version_parse_error': 'Cannot parse Python version: {output}',
                'python_version_detect_error': 'Cannot detect Python version',
                'pytorch_version_detect_error': 'Cannot detect PyTorch version',
                'pytorch_version_check_failed': 'PyTorch {version} - version check failed',
                
                # Сообщения об установке компонентов
                'pip_upgraded': 'pip upgraded successfully',
                'pip_upgrade_failed': 'pip upgrade failed: {error}',
                'triton_installed': 'Triton installed',
                'triton_failed': 'Triton failed: {error}',
                'sage_download_failed': 'Failed to download SageAttention wheel',
                'sage_installed': 'SageAttention installed',
                'sage_failed': 'SageAttention failed: {error}',
                
                # Сообщения об include/libs
                'include_libs_exist': 'include/libs folders already exist',
                'include_libs_download_failed': 'Failed to download include/libs archive',
                'include_libs_created': 'include/libs folders created',
                'include_libs_extract_no_folders': 'Extraction completed but folders not found',
                'include_libs_extract_failed': 'Extraction failed: {error}',
                
                # Системные сообщения
                'installation_cancelled': 'Installation cancelled by user',
                'critical_error': 'Critical error: {error}',
                'command_timeout': 'Command timeout',
                'python_not_found': 'Python executable not found'
            }
        }
    
    def msg(self, key: str, **kwargs) -> str:
        """
        Получить локализованное сообщение с поддержкой форматирования
        Args:
            key: ключ сообщения
            **kwargs: параметры для форматирования строки
        """
        message = self.messages[self.language].get(key, key)
        if kwargs:
            try:
                return message.format(**kwargs)
            except (KeyError, ValueError):
                return message
        return message
    
    def _run_command(self, cmd: list, timeout: int = 300) -> Tuple[bool, str]:
        """Выполнение команды с обработкой ошибок"""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8'
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, self.msg('command_timeout')
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
                return False, self.msg('python_too_old', 
                                     version=python_version, 
                                     min_version=self.MIN_PYTHON_VERSION)
            except ValueError:
                return False, self.msg('python_version_parse_error', output=output)
        return False, self.msg('python_version_detect_error')
    
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
            return False, self.msg('pytorch_version_detect_error'), None
        
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
                return False, self.msg('pytorch_version_check_failed', version=current_version), current_version
    
    def ask_pytorch_update(self, current_version: str) -> bool:
        """Запрос пользователя об обновлении PyTorch"""
        print(f"\n{self.msg('pytorch_warning')}")
        print(f"{self.msg('current_version')}: {current_version}")
        print(f"{self.msg('required_version')}: {self.MIN_PYTORCH_VERSION}+")
        print(f"{self.msg('sage_requirement', version=self.MIN_PYTORCH_VERSION)}")
        print()
        
        while True:
            choice = input(self.msg('pytorch_choice')).strip()
            
            if choice == '1':
                return True
            elif choice == '2':
                print(f"\n{self.msg('continuing_with_version', version=current_version)}")
                print(f"{self.msg('compatibility_warning')}")
                print(f"{self.msg('consider_updating')}")
                return False
            else:
                print(f"❌ {self.msg('invalid_choice')}")
    
    def install_pytorch(self) -> Tuple[bool, str]:
        """Установка PyTorch 2.7.1 с CUDA 12.8"""
        print(f"{self.msg('installing_pytorch_cuda', version=self.MIN_PYTORCH_VERSION)}")
        print(f"{self.msg('download_time_warning')}")
        
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
                return True, self.msg('pytorch_installed_success', version=new_version)
            else:
                return True, self.msg('pytorch_installed_no_check')
        else:
            return False, self.msg('pytorch_install_failed', error=output[:300])
    
    def upgrade_pip(self) -> Tuple[bool, str]:
        """Обновление pip до последней версии"""
        success, output = self._run_command([
            self.python_exe, "-m", "pip", "install", "--upgrade", "pip"
        ])
        return (success, 
                self.msg('pip_upgraded') if success 
                else self.msg('pip_upgrade_failed', error=output[:200]))
    
    def install_triton(self) -> Tuple[bool, str]:
        """Установка Triton Windows"""
        success, output = self._run_command([
            self.python_exe, "-m", "pip", "install", "-U", "triton-windows<3.4"
        ])
        return (success, 
                self.msg('triton_installed') if success 
                else self.msg('triton_failed', error=output[:200]))
    
    def install_sageattention(self) -> Tuple[bool, str]:
        """Установка SageAttention"""
        wheel_url = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/sageattention-2.2.0%2Bcu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
        wheel_file = "sageattention-2.2.0+cu128torch2.7.1.post1-cp39-abi3-win_amd64.whl"
        
        # Скачивание wheel
        if not self._download_file(wheel_url, wheel_file):
            return False, self.msg('sage_download_failed')
        
        # Установка из wheel
        success, output = self._run_command([
            self.python_exe, "-m", "pip", "install", wheel_file
        ])
        
        # Очистка
        try:
            os.remove(wheel_file)
        except:
            pass
            
        return (success, 
                self.msg('sage_installed') if success 
                else self.msg('sage_failed', error=output[:200]))
    
    def setup_include_libs(self) -> Tuple[bool, str]:
        """Скачивание и распаковка папок include/libs"""
        if os.path.exists("include") and os.path.exists("libs"):
            return True, self.msg('include_libs_exist')
        
        zip_url = "https://github.com/freyandere/TRSA-Comfyui_installer/raw/main/python_3.12.7_include_libs.zip"
        zip_file = "include_libs.zip"
        
        # Скачивание
        if not self._download_file(zip_url, zip_file):
            return False, self.msg('include_libs_download_failed')
        
        # Распаковка
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zip_file)
            
            if os.path.exists("include") and os.path.exists("libs"):
                return True, self.msg('include_libs_created')
            return False, self.msg('include_libs_extract_no_folders')
            
        except Exception as e:
            return False, self.msg('include_libs_extract_failed', error=str(e))
    
    def choose_language(self):
        """Выбор языка пользователем с пересозданием сообщений"""
        print(self.msg('lang_choice'), end='')
        try:
            choice = input().strip()
            if choice == '1':
                self.language = 'en'
                self.messages = self._load_messages()
            elif choice == '2':
                self.language = 'ru'
                self.messages = self._load_messages()
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
                    print(f"\n{self.msg('pytorch_installing')}...")
                    install_success, install_message = self.install_pytorch()
                    if install_success:
                        print(f"✅ {install_message}")
                    else:
                        print(f"❌ {install_message}")
                        print(f"⚠️ {self.msg('continuing_current_pytorch')}")
            else:
                # PyTorch не установлен вообще
                print(f"\n{self.msg('pytorch_installing')}...")
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
            print(f"✅ {self.msg('success')}")
            print(self.msg('restart_note'))
        else:
            print(f"⚠️ {self.msg('some_failed')}")
        
        return all_success


if __name__ == "__main__":
    installer = None
    try:
        installer = EnhancedSimpleInstaller()
        installer.run_installation()
    except KeyboardInterrupt:
        print(f"\n👋 {installer.msg('installation_cancelled') if installer else 'Installation cancelled by user'}")
    except Exception as e:
        error_msg = installer.msg('critical_error', error=e) if installer else f"Critical error: {e}"
        print(f"❌ {error_msg}")
    finally:
        print()
        try:
            if installer:
                print(installer.msg('press_enter'))
            else:
                print("Press Enter to exit... / Нажмите Enter для выхода...")
        except:
            print("Press Enter to exit... / Нажмите Enter для выхода...")
        input()
