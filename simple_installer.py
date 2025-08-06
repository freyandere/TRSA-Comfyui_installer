#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Accelerator v4.0 - Simple & Fixed Installation
Исправляет все проблемы и предупреждения
"""
import subprocess
import urllib.request
import zipfile
import sys
import os
import warnings
from pathlib import Path
from typing import Tuple, Dict

class SimpleInstaller:
    """Простой установщик ComfyUI Accelerator с исправленными ошибками"""
    
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
        """
        Современный способ определения языка (исправлено DeprecationWarning)
        Заменяет устаревший locale.getdefaultlocale()
        """
        try:
            # Метод 1: Переменные окружения (наиболее надежный)
            for env_var in ['LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE']:
                lang = os.environ.get(env_var, '').lower()
                if lang.startswith('ru'):
                    return 'ru'
            
            # Метод 2: Windows - проверяем реестр
            if os.name == 'nt':
                try:
                    import subprocess
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
            
            # Метод 3: Современный locale API (Python 3.11+)
            try:
                import locale
                # Используем современный API вместо устаревшего getdefaultlocale()
                current_locale = locale.getlocale()
                if current_locale[0] and current_locale[0].lower().startswith('ru'):
                    return 'ru'
                    
                # Альтернативный способ через encoding
                encoding = locale.getencoding()
                if 'ru' in encoding.lower():
                    return 'ru'
            except:
                pass
                
        except Exception:
            pass
        
        # Fallback - английский по умолчанию
        return 'en'
    
    def _load_messages(self) -> Dict[str, Dict[str, str]]:
        """Загрузка сообщений для двух языков"""
        return {
            'ru': {
                'header': '🚀 ComfyUI Accelerator - Простая установка',
                'lang_choice': 'Выберите язык / Choose language: 1-English, 2-Русский: ',
                'checking_python': 'Проверка версии Python',
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
                'header': '🚀 ComfyUI Accelerator - Simple Installation',
                'lang_choice': 'Choose language / Выберите язык: 1-English, 2-Русский: ',
                'checking_python': 'Checking Python version',
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
                version = float(output.strip())
                if version >= 3.12:
                    return True, f"Python {version} - OK"
                return False, f"Python {version} too old (need 3.12+)"
            except ValueError:
                return False, f"Cannot parse Python version: {output}"
        return False, "Cannot detect Python version"
    
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
        """Выполнение полной установки"""
        # Выбор языка
        self.choose_language()
        
        steps = [
            (self.msg('checking_python'), self.check_python_version),
            (self.msg('upgrading_pip'), self.upgrade_pip),
            (self.msg('installing_triton'), self.install_triton),
            (self.msg('installing_sage'), self.install_sageattention),
            (self.msg('setting_up_libs'), self.setup_include_libs)
        ]
        
        print(self.msg('header'))
        print("=" * 50)
        
        all_success = True
        for i, (description, method) in enumerate(steps, 1):
            print(f"\n{i}/{len(steps)} {description}...")
            success, message = method()
            
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                all_success = False
        
        print("\n" + "=" * 50)
        if all_success:
            print(f"🎉 {self.msg('success')}")
            print(self.msg('restart_note'))
        else:
            print(f"⚠️ {self.msg('some_failed')}")
        
        return all_success


if __name__ == "__main__":
    try:
        installer = SimpleInstaller()
        installer.run_installation()
    except KeyboardInterrupt:
        print("\n👋 Installation cancelled by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        print()
        # Получаем сообщение на правильном языке
        try:
            print(installer.msg('press_enter'))
        except:
            print("Press Enter to exit... / Нажмите Enter для выхода...")
        input()
