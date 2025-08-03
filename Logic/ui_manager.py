#!/usr/bin/env python3
"""
ComfyUI Accelerator v3.0 - UI Manager Module
Handles multilingual interface and user interactions
"""

import os
import sys
import logging
import platform
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

class Language(Enum):
    """Supported languages"""
    RUSSIAN = "ru"
    ENGLISH = "en"

@dataclass
class UITheme:
    """UI theme configuration"""
    progress_char: str = "█"
    empty_char: str = "░"
    success_icon: str = "✅"
    error_icon: str = "❌"
    warning_icon: str = "⚠️"
    info_icon: str = "ℹ️"
    loading_icons: List[str] = None
    
    def __post_init__(self):
        if self.loading_icons is None:
            self.loading_icons = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

class UIManager:
    """Multilingual UI manager with progress indicators and menus"""
    
    def __init__(self, language: Language = None, theme: UITheme = None):
        self.logger = logging.getLogger(__name__)
        self.theme = theme or UITheme()
        self.language = language or self._detect_system_language()
        self._load_translations()
        
    def _detect_system_language(self) -> Language:
        """Auto-detect system language"""
        try:
            if os.name == 'nt':  # Windows
                import subprocess
                result = subprocess.run([
                    'reg', 'query', 'HKCU\\Control Panel\\International', 
                    '/v', 'LocaleName'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'LocaleName' in line:
                            locale = line.split()[-1]
                            if locale.startswith('ru'):
                                return Language.RUSSIAN
            
            # Fallback to environment variables
            lang = os.environ.get('LANG', os.environ.get('LC_ALL', 'en'))
            if lang.startswith('ru'):
                return Language.RUSSIAN
                
        except Exception as e:
            self.logger.debug(f"Language detection failed: {e}")
        
        return Language.ENGLISH
    
    def _load_translations(self):
        """Load translation strings"""
        self.translations = {
            Language.RUSSIAN: {
                # General
                'app_title': 'УСКОРИТЕЛЬ COMFYUI v3.0',
                'app_subtitle': 'Делает генерацию в 2-3 раза быстрее!',
                'loading': 'Загрузка...',
                'please_wait': 'Пожалуйста, подождите...',
                'completed': 'Завершено!',
                'error': 'Ошибка',
                'success': 'Успех',
                'warning': 'Предупреждение',
                'info': 'Информация',
                'yes': 'Да',
                'no': 'Нет',
                'continue': 'Продолжить',
                'exit': 'Выход',
                'back': 'Назад',
                
                # Menu items
                'menu_smart_install': '🚀 УСКОРИТЬ МОЙ COMFYUI',
                'menu_smart_install_desc': '(Всё автоматически)',
                'menu_check_install': '🔍 Проверить установку',
                'menu_check_install_desc': '(Узнать что работает)',
                'menu_reinstall': '🛠️ Переустановить',
                'menu_reinstall_desc': '(Если что-то пошло не так)',
                'menu_detailed_report': '📊 Детальный отчёт',
                'menu_detailed_report_desc': '(Подробная диагностика)',
                'menu_exit': '❌ Выход',
                
                # Installation messages
                'install_checking': 'Проверяю систему...',
                'install_upgrading_pip': 'Обновляю pip...',
                'install_setup_folders': 'Настраиваю папки include/libs...',
                'install_triton': 'Устанавливаю Triton...',
                'install_sageattention': 'Устанавливаю SageAttention...',
                'install_verifying': 'Проверяю установку...',
                'install_complete': '✅ Готово! ComfyUI ускорен в 2-3 раза!',
                'install_restart_note': '💡 Перезапустите ComfyUI для применения изменений',
                
                # System info
                'system_info': 'Информация о системе',
                'python_version': 'Версия Python',
                'gpu_info': 'Информация о GPU',
                'components_status': 'Статус компонентов',
                
                # Prompts
                'choice_prompt': 'Ваш выбор: ',
                'confirm_reinstall': 'Вы уверены, что хотите переустановить всё? (y/n): ',
                'press_any_key': 'Нажмите любую клавишу для продолжения...',
                
                # Status messages
                'triton_installed': 'Triton установлен',
                'triton_not_installed': 'Triton не установлен',
                'sageattention_installed': 'SageAttention установлен',
                'sageattention_not_installed': 'SageAttention не установлен',
                'folders_ok': 'Обе папки include/libs найдены',
                'folders_missing': 'Одна или обе папки include/libs отсутствуют',
                'pytorch_ok': 'PyTorch работает',
                'pytorch_missing': 'PyTorch не найден',
                'cuda_available': 'CUDA доступна',
                'cuda_not_available': 'CUDA недоступна',
                
                # Health scores
                'health_excellent': 'Отлично',
                'health_very_good': 'Очень хорошо',
                'health_good': 'Хорошо',
                'health_fair': 'Удовлетворительно',
                'health_poor': 'Плохо',
                'health_critical': 'Критические проблемы',
                
                # Errors
                'error_python_not_found': 'Python не найден! Поместите скрипт в папку с python.exe',
                'error_download_failed': 'Ошибка загрузки',
                'error_installation_failed': 'Ошибка установки',
                'error_unexpected': 'Неожиданная ошибка',
                
                'goodbye': '👋 До свидания!'
            },
            
            Language.ENGLISH: {
                # General
                'app_title': 'COMFYUI ACCELERATOR v3.0',
                'app_subtitle': 'Makes ComfyUI 2-3x faster!',
                'loading': 'Loading...',
                'please_wait': 'Please wait...',
                'completed': 'Completed!',
                'error': 'Error',
                'success': 'Success',
                'warning': 'Warning',
                'info': 'Info',
                'yes': 'Yes',
                'no': 'No',
                'continue': 'Continue',
                'exit': 'Exit',
                'back': 'Back',
                
                # Menu items
                'menu_smart_install': '🚀 SPEED UP MY COMFYUI',
                'menu_smart_install_desc': '(Fully automatic)',
                'menu_check_install': '🔍 Check Installation',
                'menu_check_install_desc': '(See what\'s working)',
                'menu_reinstall': '🛠️ Reinstall Everything',
                'menu_reinstall_desc': '(If something went wrong)',
                'menu_detailed_report': '📊 Detailed Report',
                'menu_detailed_report_desc': '(Full diagnostics)',
                'menu_exit': '❌ Exit',
                
                # Installation messages
                'install_checking': 'Checking your system...',
                'install_upgrading_pip': 'Upgrading pip...',
                'install_setup_folders': 'Setting up include/libs folders...',
                'install_triton': 'Installing Triton...',
                'install_sageattention': 'Installing SageAttention...',
                'install_verifying': 'Verifying installation...',
                'install_complete': '✅ Done! ComfyUI is now 2-3x faster!',
                'install_restart_note': '💡 Restart ComfyUI to apply changes',
                
                # System info
                'system_info': 'System Information',
                'python_version': 'Python Version',
                'gpu_info': 'GPU Information',
                'components_status': 'Components Status',
                
                # Prompts
                'choice_prompt': 'Your choice: ',
                'confirm_reinstall': 'Are you sure you want to reinstall everything? (y/n): ',
                'press_any_key': 'Press any key to continue...',
                
                # Status messages
                'triton_installed': 'Triton installed',
                'triton_not_installed': 'Triton not installed',
                'sageattention_installed': 'SageAttention installed',
                'sageattention_not_installed': 'SageAttention not installed',
                'folders_ok': 'Both include/libs folders found',
                'folders_missing': 'One or both include/libs folders are missing',
                'pytorch_ok': 'PyTorch working',
                'pytorch_missing': 'PyTorch not found',
                'cuda_available': 'CUDA available',
                'cuda_not_available': 'CUDA not available',
                
                # Health scores
                'health_excellent': 'Excellent',
                'health_very_good': 'Very Good',
                'health_good': 'Good',
                'health_fair': 'Fair',
                'health_poor': 'Poor',
                'health_critical': 'Critical Issues',
                
                # Errors
                'error_python_not_found': 'Python not found! Place script in folder with python.exe',
                'error_download_failed': 'Download failed',
                'error_installation_failed': 'Installation failed',
                'error_unexpected': 'Unexpected error',
                
                'goodbye': '👋 Goodbye!'
            }
        }
    
    def t(self, key: str, **kwargs) -> str:
        """Get translated string"""
        text = self.translations[self.language].get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except:
                pass
        return text
    
    def set_language(self, language: Language):
        """Change interface language"""
        self.language = language
        self.logger.info(f"Language changed to {language.value}")
    
    def clear_screen(self):
        """Clear console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str = None):
        """Print application header"""
        self.clear_screen()
        print()
        print("=" * 60)
        print(f"  {title or self.t('app_title')}")
        print(f"  {self.t('app_subtitle')}")
        print("=" * 60)
        print()
    
    def print_divider(self, char: str = "=", length: int = 60):
        """Print divider line"""
        print(char * length)
    
    def show_menu(self, items: List[Tuple[str, str, Callable]], title: str = None) -> bool:
        """Show interactive menu and handle selection"""
        self.print_header(title)
        
        # Display menu items
        for i, (label, description, _) in enumerate(items, 1):
            print(f"{i}. {label}")
            if description:
                print(f"   {description}")
            print()
        
        # Get user choice
        try:
            choice = input(self.t('choice_prompt')).strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    _, _, callback = items[idx]
                    if callback:
                        return callback()
            elif choice.lower() in ['e', 'exit', 'q', 'quit']:
                return False
                
        except KeyboardInterrupt:
            print(f"\n{self.t('goodbye')}")
            return False
        except Exception as e:
            self.show_error(f"{self.t('error_unexpected')}: {str(e)}")
        
        return True
    
    def show_progress_bar(self, current: int, total: int, description: str = "", width: int = 40) -> str:
        """Generate progress bar string"""
        if total <= 0:
            return f"[{'?' * width}] 0% {description}"
        
        percentage = min(100, (current * 100) // total)
        filled = min(width, (current * width) // total)
        empty = width - filled
        
        bar = (self.theme.progress_char * filled) + (self.theme.empty_char * empty)
        return f"[{bar}] {percentage}% {description}"
    
    def show_step_progress(self, step: int, total_steps: int, description: str):
        """Show step-by-step progress"""
        progress_bar = self.show_progress_bar(step, total_steps, description)
        print(f"\r{progress_bar}", end="", flush=True)
        
        if step >= total_steps:
            print()  # New line when complete
    
    def show_spinner(self, message: str = "", duration: float = 1.0):
        """Show animated spinner"""
        import time
        import itertools
        
        spinner = itertools.cycle(self.theme.loading_icons)
        end_time = time.time() + duration
        
        while time.time() < end_time:
            print(f"\r{next(spinner)} {message}", end="", flush=True)
            time.sleep(0.1)
        
        print(f"\r{self.theme.success_icon} {message}", flush=True)
    
    def show_success(self, message: str):
        """Show success message"""
        print(f"{self.theme.success_icon} {message}")
    
    def show_error(self, message: str):
        """Show error message"""
        print(f"{self.theme.error_icon} {message}")
    
    def show_warning(self, message: str):
        """Show warning message"""
        print(f"{self.theme.warning_icon} {message}")
    
    def show_info(self, message: str):
        """Show info message"""
        print(f"{self.theme.info_icon} {message}")
    
    def confirm(self, message: str) -> bool:
        """Show confirmation dialog"""
        try:
            response = input(f"{message} ").strip().lower()
            return response in ['y', 'yes', 'да', 'д']
        except KeyboardInterrupt:
            return False
    
    def wait_for_input(self, message: str = None):
        """Wait for user input"""
        try:
            input(message or self.t('press_any_key'))
        except KeyboardInterrupt:
            pass
    
    def show_system_info(self, system_info: Dict[str, Any]):
        """Display system information in formatted way"""
        print(f"🐍 {self.t('python_version')}: {system_info.get('python_version', 'Unknown')}")
        print(f"📦 pip: {system_info.get('pip_version', 'Unknown')}")
        print(f"💻 Platform: {system_info.get('platform_info', 'Unknown')}")
        
        if system_info.get('pytorch_version'):
            print(f"🧠 PyTorch: {system_info['pytorch_version']}")
            
        if system_info.get('cuda_available'):
            print(f"🔥 CUDA: {self.t('cuda_available')}")
            if system_info.get('cuda_version'):
                print(f"   Version: {system_info['cuda_version']}")
        else:
            print(f"🔥 CUDA: {self.t('cuda_not_available')}")
            
        if system_info.get('gpu_name'):
            print(f"🎮 GPU: {system_info['gpu_name']}")
            if system_info.get('gpu_driver'):
                print(f"   Driver: {system_info['gpu_driver']}")
    
    def show_component_status(self, components: Dict[str, Any]):
        """Display component installation status"""
        status_map = {
            'triton': ('triton_installed', 'triton_not_installed'),
            'sageattention': ('sageattention_installed', 'sageattention_not_installed'),
            'pytorch': ('pytorch_ok', 'pytorch_missing')
        }
        
        for component, (ok_key, fail_key) in status_map.items():
            if component in components:
                if components[component]:
                    self.show_success(self.t(ok_key))
                else:
                    self.show_error(self.t(fail_key))
        
        # Check folders (support both 'folders' and separate keys)
        include_ok = components.get('include_folder', components.get('folders', False))
        libs_ok = components.get('libs_folder', components.get('folders', False))
        
        if include_ok and libs_ok:
            self.show_success(self.t('folders_ok'))
        else:
            self.show_error(self.t('folders_missing'))
        
        # Check folders
        if components.get('include_folder') and components.get('libs_folder'):
            self.show_success(self.t('folders_ok'))
        else:
            self.show_error(self.t('folders_missing'))
    
    def show_health_score(self, health_data: Dict[str, Any]):
        """Display system health score"""
        score = health_data.get('percentage', 0)
        grade = health_data.get('grade', 'Unknown')
        
        print(f"\n🎯 Health Score: {score}% ({grade})")
        
        if health_data.get('issues'):
            print(f"\n{self.theme.warning_icon} Issues found:")
            for issue in health_data['issues']:
                print(f"   • {issue}")
    
    def language_selection_menu(self) -> Language:
        """Show language selection menu"""
        self.clear_screen()
        print("\n" + "=" * 40)
        print("  Language Selection / Выбор языка")  
        print("=" * 40)
        print()
        print("1. English")
        print("2. Русский")
        print()
        
        while True:
            try:
                choice = input("Select language / Выберите язык (1-2): ").strip()
                if choice == "1":
                    return Language.ENGLISH
                elif choice == "2":
                    return Language.RUSSIAN
                else:
                    print("Invalid choice / Неверный выбор")
            except KeyboardInterrupt:
                return Language.ENGLISH

if __name__ == "__main__":
    # Demo of UI Manager
    ui = UIManager()
    
    # Language selection
    lang = ui.language_selection_menu()
    ui.set_language(lang)
    
    # Demo menu
    def demo_install():
        ui.print_header()
        for i in range(1, 6):
            ui.show_step_progress(i, 5, ui.t('install_checking'))
            __import__('time').sleep(0.5)
        ui.show_success(ui.t('install_complete'))
        ui.wait_for_input()
        return True
    
    def demo_check():
        ui.print_header("System Check")
        ui.show_info("Running system diagnostics...")
        ui.show_spinner(ui.t('please_wait'), 2.0)
        ui.show_success("All systems operational!")
        ui.wait_for_input()
        return True
    
    def demo_exit():
        print(ui.t('goodbye'))
        return False
    
    # Main menu
    menu_items = [
        (ui.t('menu_smart_install'), ui.t('menu_smart_install_desc'), demo_install),
        (ui.t('menu_check_install'), ui.t('menu_check_install_desc'), demo_check),
        (ui.t('menu_exit'), "", demo_exit)
    ]
    
    while ui.show_menu(menu_items):
        pass



