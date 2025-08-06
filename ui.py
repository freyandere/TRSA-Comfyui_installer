#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRSA ComfyUI Accelerator v5.1 - UI & Localization (Temporary Module)
Управление локализацией и пользовательским интерфейсом для bat-запуска
"""
import os
import subprocess
import locale
import sys
from typing import Dict, Any, Optional

# Импорт config с обработкой временных имён модулей
try:
    if 'temp_config' in sys.modules:
        from temp_config import Config
    else:
        # Попытка импорта как временного модуля
        import importlib.util
        spec = importlib.util.spec_from_file_location("temp_config", "temp_config.py")
        if spec and spec.loader:
            temp_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_config)
            Config = temp_config.Config
        else:
            from config import Config
except ImportError:
    # Fallback для development
    try:
        from config import Config
    except ImportError:
        # Создаём минимальную конфигурацию если ничего не найдено
        class Config:
            APP_VERSION = "5.1"
            TARGET_PYTORCH_VERSION = "2.7.1"
            MAX_PYTORCH_VERSION = "2.7.9"
            LOCALE_ENV_VARS = ['LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE']
            REGISTRY_LOCALE_PATH = 'HKCU\\Control Panel\\International'
            REGISTRY_TIMEOUT = 5


class LocalizationManager:
    """Менеджер локализации с автоопределением языка для bat-контекста"""
    
    def __init__(self):
        self.language = self._detect_language()
        self._messages = self._load_all_messages()
    
    def _detect_language(self) -> str:
        """Определение языка системы с улучшенной обработкой для Windows"""
        try:
            # 1. Проверка переменных окружения
            for env_var in Config.LOCALE_ENV_VARS:
                lang = os.environ.get(env_var, '').lower()
                if lang.startswith('ru'):
                    return 'ru'
            
            # 2. Windows Registry (оптимизировано для bat-контекста)
            if os.name == 'nt':
                try:
                    result = subprocess.run([
                        'reg', 'query', Config.REGISTRY_LOCALE_PATH,
                        '/v', 'LocaleName'
                    ], capture_output=True, text=True, 
                    timeout=Config.REGISTRY_TIMEOUT, shell=True)
                    
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'LocaleName' in line and 'ru' in line.lower():
                                return 'ru'
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    # В bat-контексте могут быть проблемы с PATH
                    pass
            
            # 3. Python locale API с дополнительной обработкой ошибок
            try:
                current_locale = locale.getlocale()
                if current_locale[0] and current_locale[0].lower().startswith('ru'):
                    return 'ru'
                    
                # Проверка кодировки
                try:
                    encoding = locale.getencoding()
                    if 'ru' in encoding.lower():
                        return 'ru'
                except AttributeError:
                    # На старых версиях Python getencoding() может отсутствовать
                    pass
                    
            except (AttributeError, locale.Error):
                pass
            
            # 4. Проверка системной кодировки Windows
            if os.name == 'nt':
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    lcid = kernel32.GetUserDefaultLCID()
                    if lcid in (1049, 2073):  # Russian LCID codes
                        return 'ru'
                except (ImportError, AttributeError, OSError):
                    pass
                
        except Exception:
            pass
        
        return 'en'  # Default to English
    
    def _load_all_messages(self) -> Dict[str, Dict[str, str]]:
        """Загрузка всех локализованных сообщений"""
        return {
            'ru': self._get_russian_messages(),
            'en': self._get_english_messages()
        }
    
    def _get_russian_messages(self) -> Dict[str, str]:
        """Русские сообщения (полная версия)"""
        return {
            # Заголовки и основной интерфейс
            'app_title': f'TRSA ComfyUI Accelerator v{Config.APP_VERSION} - Умная установка',
            'lang_choice': 'Выберите язык / Choose language: 1-English, 2-Русский: ',
            'separator': '=' * 60,
            'press_enter': 'Нажмите Enter для выхода...',
            'bat_context_detected': 'Обнаружен запуск через bat-файл',
            
            # Этапы установки
            'step_checking_python': 'Проверка версии Python',
            'step_checking_pytorch': 'Проверка версии PyTorch', 
            'step_upgrading_pip': 'Обновление pip',
            'step_installing_triton': 'Установка Triton',
            'step_installing_sage': 'Установка SageAttention',
            'step_setting_up_libs': 'Настройка include/libs',
            
            # Статусы PyTorch
            'pytorch_compatible': 'PyTorch версия совместима',
            'pytorch_missing': 'PyTorch не найден - требуется установка',
            'pytorch_outdated': 'PyTorch устарел - рекомендуется обновление',
            'pytorch_too_new': 'PyTorch слишком новый - рекомендуется даунгрейд',
            
            # Предупреждения
            'warning_old_version': 'ВНИМАНИЕ: Старая версия может вызвать ошибки!',
            'warning_new_version': 'ВНИМАНИЕ: Более новая версия может вызвать проблемы совместимости!',
            'compatibility_warning_old': 'Предупреждение: Могут возникнуть проблемы совместимости!',
            'compatibility_warning_new': 'Предупреждение: Новая версия может быть несовместима с SageAttention2!',
            
            # Информация о версиях
            'current_version': 'Текущая версия',
            'target_version': 'Рекомендуемая версия',
            'compatible_range': 'Совместимый диапазон',
            'sage_requirement': 'SageAttention2 требует PyTorch {target} для оптимальной производительности',
            
            # Выборы пользователя
            'choice_update_pytorch': '1-Обновить PyTorch, 2-Продолжить с текущей версией: ',
            'choice_downgrade_pytorch': '1-Откатиться до PyTorch {target}, 2-Продолжить с текущей версией: ',
            'invalid_choice': 'Пожалуйста, введите 1 или 2',
            
            # Процесс установки
            'installing_pytorch': 'Установка PyTorch {version}',
            'downgrading_pytorch': 'Даунгрейд PyTorch до версии {version}',
            'installing_pytorch_cuda': 'Установка PyTorch {version} с CUDA 12.8...',
            'download_warning': 'Это может занять несколько минут (~2.5GB загрузка)',
            'continuing_with_version': 'Продолжаем с PyTorch {version}',
            'continuing_current_pytorch': 'Продолжаем с текущей версией PyTorch...',
            
            # Результаты установки
            'pytorch_installed_success': 'PyTorch {version} установлен успешно',
            'pytorch_installed_no_check': 'PyTorch установлен (проверка версии не удалась)',
            'pytorch_install_failed': 'Установка PyTorch не удалась: {error}',
            'pip_upgraded': 'pip обновлен успешно',
            'pip_upgrade_failed': 'Обновление pip не удалось: {error}',
            'triton_installed': 'Triton установлен',
            'triton_failed': 'Установка Triton не удалась: {error}',
            'sage_installed': 'SageAttention установлен',
            'sage_failed': 'Установка SageAttention не удалась: {error}',
            'sage_download_failed': 'Не удалось скачать wheel файл SageAttention',
            
            # include/libs
            'libs_exist': 'Папки include/libs уже существуют',
            'libs_created': 'Папки include/libs созданы',
            'libs_download_failed': 'Не удалось скачать архив include/libs',
            'libs_extract_failed': 'Распаковка не удалась: {error}',
            'libs_extract_no_folders': 'Распаковка завершена, но папки не найдены',
            
            # Ошибки Python
            'python_too_old': 'Python {version} слишком старый (требуется {min_version}+)',
            'python_parse_error': 'Не удается распознать версию Python: {output}',
            'python_detect_error': 'Не удается определить версию Python',
            'pytorch_detect_error': 'Не удается определить версию PyTorch',
            'pytorch_check_failed': 'PyTorch {version} - проверка версии не удалась',
            
            # Системные сообщения
            'installation_cancelled': 'Установка отменена пользователем',
            'critical_error': 'Критическая ошибка: {error}',
            'command_timeout': 'Превышено время ожидания команды',
            'python_not_found': 'Исполняемый файл Python не найден',
            
            # Финальные сообщения
            'installation_success': 'Установка завершена успешно!',
            'installation_partial': 'Некоторые компоненты не установились',
            'restart_note': 'Перезапустите ComfyUI для применения ускорения',
            
            # Советы
            'consider_updating': 'Рассмотрите возможность обновления PyTorch позже при возникновении проблем',
            'consider_downgrading': 'Рассмотрите возможность отката PyTorch при возникновении проблем'
        }
    
    def _get_english_messages(self) -> Dict[str, str]:
        """English messages (полная версия)"""
        return {
            # Headers and main interface
            'app_title': f'TRSA ComfyUI Accelerator v{Config.APP_VERSION} - Smart Installation',
            'lang_choice': 'Choose language / Выберите язык: 1-English, 2-Русский: ',
            'separator': '=' * 60,
            'press_enter': 'Press Enter to exit...',
            'bat_context_detected': 'Bat-file execution context detected',
            
            # Installation steps
            'step_checking_python': 'Checking Python version',
            'step_checking_pytorch': 'Checking PyTorch version',
            'step_upgrading_pip': 'Upgrading pip',
            'step_installing_triton': 'Installing Triton',
            'step_installing_sage': 'Installing SageAttention',
            'step_setting_up_libs': 'Setting up include/libs',
            
            # PyTorch statuses
            'pytorch_compatible': 'PyTorch version is compatible',
            'pytorch_missing': 'PyTorch not found - installation required',
            'pytorch_outdated': 'PyTorch is outdated - update recommended',
            'pytorch_too_new': 'PyTorch is too new - downgrade recommended',
            
            # Warnings
            'warning_old_version': 'WARNING: Old version may cause errors!',
            'warning_new_version': 'WARNING: Newer version may cause compatibility issues!',
            'compatibility_warning_old': 'Warning: You may encounter compatibility issues!',
            'compatibility_warning_new': 'Warning: Newer version may be incompatible with SageAttention2!',
            
            # Version information
            'current_version': 'Current version',
            'target_version': 'Recommended version',
            'compatible_range': 'Compatible range',
            'sage_requirement': 'SageAttention2 requires PyTorch {target} for optimal performance',
            
            # User choices
            'choice_update_pytorch': '1-Update PyTorch, 2-Continue with current version: ',
            'choice_downgrade_pytorch': '1-Downgrade to PyTorch {target}, 2-Continue with current version: ',
            'invalid_choice': 'Please enter 1 or 2',
            
            # Installation process
            'installing_pytorch': 'Installing PyTorch {version}',
            'downgrading_pytorch': 'Downgrading PyTorch to version {version}',
            'installing_pytorch_cuda': 'Installing PyTorch {version} with CUDA 12.8...',
            'download_warning': 'This may take several minutes (~2.5GB download)',
            'continuing_with_version': 'Continuing with PyTorch {version}',
            'continuing_current_pytorch': 'Continuing with current PyTorch version...',
            
            # Installation results
            'pytorch_installed_success': 'PyTorch {version} installed successfully',
            'pytorch_installed_no_check': 'PyTorch installed (version check failed)',
            'pytorch_install_failed': 'PyTorch installation failed: {error}',
            'pip_upgraded': 'pip upgraded successfully',
            'pip_upgrade_failed': 'pip upgrade failed: {error}',
            'triton_installed': 'Triton installed',
            'triton_failed': 'Triton failed: {error}',
            'sage_installed': 'SageAttention installed',
            'sage_failed': 'SageAttention failed: {error}',
            'sage_download_failed': 'Failed to download SageAttention wheel',
            
            # include/libs
            'libs_exist': 'include/libs folders already exist',
            'libs_created': 'include/libs folders created',
            'libs_download_failed': 'Failed to download include/libs archive',
            'libs_extract_failed': 'Extraction failed: {error}',
            'libs_extract_no_folders': 'Extraction completed but folders not found',
            
            # Python errors
            'python_too_old': 'Python {version} too old (need {min_version}+)',
            'python_parse_error': 'Cannot parse Python version: {output}',
            'python_detect_error': 'Cannot detect Python version',
            'pytorch_detect_error': 'Cannot detect PyTorch version',
            'pytorch_check_failed': 'PyTorch {version} - version check failed',
            
            # System messages
            'installation_cancelled': 'Installation cancelled by user',
            'critical_error': 'Critical error: {error}',
            'command_timeout': 'Command timeout',
            'python_not_found': 'Python executable not found',
            
            # Final messages
            'installation_success': 'Installation completed successfully!',
            'installation_partial': 'Some components failed to install',
            'restart_note': 'Restart ComfyUI to apply 2-3x speed improvements',
            
            # Tips
            'consider_updating': 'Consider updating PyTorch later if problems occur',
            'consider_downgrading': 'Consider downgrading PyTorch if problems occur'
        }
    
    def get_message(self, key: str, **kwargs) -> str:
        """
        Получить локализованное сообщение с форматированием
        
        Args:
            key: ключ сообщения
            **kwargs: параметры для форматирования
            
        Returns:
            Отформатированное сообщение на текущем языке
        """
        message = self._messages[self.language].get(key, key)
        
        if kwargs:
            try:
                return message.format(**kwargs)
            except (KeyError, ValueError):
                return message
        
        return message
    
    def set_language(self, language: str) -> None:
        """Установить язык интерфейса"""
        if language in self._messages:
            self.language = language
    
    def ask_language_choice(self) -> str:
        """Интерактивный выбор языка пользователем"""
        print(self.get_message('lang_choice'), end='')
        
        try:
            choice = input().strip()
            if choice == '1':
                self.set_language('en')
            elif choice == '2':
                self.set_language('ru')
        except (EOFError, KeyboardInterrupt):
            pass  # Оставляем автоопределенный язык
        
        return self.language


class UserInterface:
    """Класс для управления пользовательским интерфейсом в bat-контексте"""
    
    def __init__(self, localization: LocalizationManager):
        self.loc = localization
        self._is_bat_context = self._detect_bat_context()
    
    def _detect_bat_context(self) -> bool:
        """Определение запуска через bat-файл"""
        return (
            'temp_' in __file__ or 
            any('temp_' in name for name in sys.modules.keys()) or
            os.path.exists('temp_main.py')
        )
    
    def print_header(self) -> None:
        """Вывод заголовка приложения"""
        print(self.loc.get_message('app_title'))
        print(self.loc.get_message('separator'))
        
        if self._is_bat_context:
            print(f"🔄 {self.loc.get_message('bat_context_detected')}")
            print()
    
    def print_step(self, step: int, total: int, message_key: str) -> None:
        """Вывод информации о текущем шаге"""
        print(f"\n{step}/{total} {self.loc.get_message(message_key)}...")
    
    def print_success(self, message: str) -> None:
        """Вывод сообщения об успехе"""
        print(f"✅ {message}")
    
    def print_error(self, message: str) -> None:
        """Вывод сообщения об ошибке"""
        print(f"❌ {message}")
    
    def print_warning(self, message: str) -> None:
        """Вывод предупреждения"""
        print(f"⚠️ {message}")
    
    def ask_pytorch_action(self, current_version: str, action_type: str) -> bool:
        """Запрос действия пользователя относительно PyTorch"""
        # Определяем тип предупреждения и выбора
        if action_type == 'update':
            warning_key = 'warning_old_version'
            choice_key = 'choice_update_pytorch'
            compatibility_key = 'compatibility_warning_old'
            advice_key = 'consider_updating'
        else:  # downgrade
            warning_key = 'warning_new_version' 
            choice_key = 'choice_downgrade_pytorch'
            compatibility_key = 'compatibility_warning_new'
            advice_key = 'consider_downgrading'
        
        # Выводим информацию
        print(f"\n{self.loc.get_message(warning_key)}")
        print(f"{self.loc.get_message('current_version')}: {current_version}")
        print(f"{self.loc.get_message('target_version')}: {Config.TARGET_PYTORCH_VERSION}")
        
        if action_type == 'downgrade':
            print(f"{self.loc.get_message('compatible_range')}: {Config.TARGET_PYTORCH_VERSION} - {Config.MAX_PYTORCH_VERSION}")
        
        print(f"{self.loc.get_message('sage_requirement', target=Config.TARGET_PYTORCH_VERSION)}")
        print()
        
        # Запрашиваем выбор
        while True:
            if action_type == 'update':
                choice = input(self.loc.get_message(choice_key)).strip()
            else:
                choice = input(self.loc.get_message(choice_key, target=Config.TARGET_PYTORCH_VERSION)).strip()
            
            if choice == '1':
                return True
            elif choice == '2':
                print(f"\n{self.loc.get_message('continuing_with_version', version=current_version)}")
                print(f"{self.loc.get_message(compatibility_key)}")
                print(f"{self.loc.get_message(advice_key)}")
                return False
            else:
                self.print_error(self.loc.get_message('invalid_choice'))
    
    def print_final_result(self, success: bool) -> None:
        """Вывод финального результата установки"""
        print(f"\n{self.loc.get_message('separator')}")
        
        if success:
            self.print_success(self.loc.get_message('installation_success'))
            print(self.loc.get_message('restart_note'))
        else:
            self.print_warning(self.loc.get_message('installation_partial'))
        
        if self._is_bat_context:
            print(f"\n🧹 Временные файлы будут удалены bat-скриптом")
    
    def wait_for_exit(self) -> None:
        """Ожидание нажатия Enter для выхода (адаптировано для bat)"""
        print()
        try:
            print(self.loc.get_message('press_enter'))
        except:
            print("Press Enter to exit... / Нажмите Enter для выхода...")
        
        # В bat-контексте не вызываем input(), так как bat-файл сам делает pause
        if not self._is_bat_context:
            input()


# Проверка корректности импорта в контексте bat-запуска
if __name__ == "__main__":
    print("UI module loaded successfully")
    loc = LocalizationManager()
    ui = UserInterface(loc)
    print(f"Language detected: {loc.language}")
    print(f"Bat context: {ui._is_bat_context}")
