"""
TRSA ComfyUI Installer - Localization Module
Version: 2.6.0
Supported languages: English, Russian
"""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, Any]] = {
    "en": {
        # Welcome screen
        "welcome_title": "TRSA ComfyUI SageAttention Installer",
        "welcome_version": "Version: {version}",
        "welcome_separator": "=" * 60,
        
        # Language selection
        "lang_select_prompt": "Select language / Выберите язык:",
        "lang_option_en": "[1] English",
        "lang_option_ru": "[2] Русский",
        "lang_default": "[Press Enter for system default]",
        "lang_choice_prompt": "Your choice: ",
        "lang_invalid": "Invalid choice. Using system default.",
        "lang_selected": "Language selected: English",
        
        # System check
        "check_title": "\nSystem Check",
        "check_python": "Python version: {version}",
        "check_torch": "PyTorch version: {version}",
        "check_cuda": "CUDA version: {version}",
        "check_sage_installed": "SageAttention: Installed (v{version})",
        "check_sage_not_installed": "SageAttention: Not installed",
        "check_compatible": "✓ Your configuration is compatible",
        "check_upgrade_needed": "⚠ Upgrade recommended",
        "check_current_config": "Current: torch={torch}, CUDA={cuda}",
        "check_target_config": "Target: torch={torch}, CUDA={cuda}",
        
        # Torch upgrade prompt
        "torch_upgrade_title": "\nPyTorch Upgrade Available",
        "torch_upgrade_msg": "Current version: torch {current} + CUDA {cuda}",
        "torch_upgrade_recommend": "Recommended: torch {target} + CUDA {cuda_target}",
        "torch_upgrade_prompt": "Upgrade PyTorch? [Y/n]: ",
        "torch_upgrade_yes": "Starting PyTorch upgrade...",
        "torch_upgrade_skip": "Skipping PyTorch upgrade",
        
        # Installation
        "install_title": "\nInstallation Process",
        "install_selecting_wheel": "Selecting wheel for torch={torch}, CUDA={cuda}",
        "install_wheel_found": "✓ Found: {wheel}",
        "install_wheel_not_found": "✗ No compatible wheel found",
        "install_downloading": "Downloading {file}...",
        "install_installing": "Installing SageAttention...",
        "install_success": "✓ Installation successful",
        "install_failed": "✗ Installation failed",
        "install_torch_upgrading": "Upgrading PyTorch to {version}...",
        
        # Rollback
        "rollback_title": "\nRollback Options",
        "rollback_prompt": "Installation failed. Rollback to previous version? [Y/n]: ",
        "rollback_starting": "Starting rollback...",
        "rollback_success": "✓ Rollback successful",
        "rollback_failed": "✗ Rollback failed",
        "rollback_skipped": "Rollback skipped",
        
        # Summary
        "summary_title": "\nInstallation Summary",
        "summary_success": "✓ Installation completed successfully",
        "summary_failed": "✗ Installation failed",
        "summary_previous_version": "Previous version: {version}",
        "summary_installed_version": "Installed version: {version}",
        "summary_torch_version": "PyTorch: {version}",
        "summary_cuda_version": "CUDA: {version}",
        "summary_errors": "Errors encountered: {count}",
        "summary_log_saved": "Detailed log saved: {path}",
        
        # Errors
        "error_not_in_python_embeded": "ERROR: This script must be run from ComfyUI's python_embeded folder",
        "error_python_not_found": "ERROR: Python executable not found",
        "error_torch_not_installed": "ERROR: PyTorch is not installed",
        "error_network": "ERROR: Network connection failed",
        "error_download_failed": "ERROR: Failed to download {file}",
        "error_install_failed": "ERROR: Installation failed - {reason}",
        
        # Cleanup
        "cleanup_title": "\nCleanup",
        "cleanup_removing": "Removing temporary files...",
        "cleanup_success": "✓ Cleanup completed",
        
        # Prompts
        "press_enter": "\nPress Enter to exit...",
    },
    
    "ru": {
        # Welcome screen
        "welcome_title": "TRSA ComfyUI SageAttention Установщик",
        "welcome_version": "Версия: {version}",
        "welcome_separator": "=" * 60,
        
        # Language selection
        "lang_select_prompt": "Выберите язык / Select language:",
        "lang_option_en": "[1] English",
        "lang_option_ru": "[2] Русский",
        "lang_default": "[Нажмите Enter для системного по умолчанию]",
        "lang_choice_prompt": "Ваш выбор: ",
        "lang_invalid": "Неверный выбор. Используется системный язык.",
        "lang_selected": "Выбран язык: Русский",
        
        # System check
        "check_title": "\nПроверка системы",
        "check_python": "Версия Python: {version}",
        "check_torch": "Версия PyTorch: {version}",
        "check_cuda": "Версия CUDA: {version}",
        "check_sage_installed": "SageAttention: Установлен (v{version})",
        "check_sage_not_installed": "SageAttention: Не установлен",
        "check_compatible": "✓ Ваша конфигурация совместима",
        "check_upgrade_needed": "⚠ Рекомендуется обновление",
        "check_current_config": "Текущая: torch={torch}, CUDA={cuda}",
        "check_target_config": "Целевая: torch={torch}, CUDA={cuda}",
        
        # Torch upgrade prompt
        "torch_upgrade_title": "\nДоступно обновление PyTorch",
        "torch_upgrade_msg": "Текущая версия: torch {current} + CUDA {cuda}",
        "torch_upgrade_recommend": "Рекомендуется: torch {target} + CUDA {cuda_target}",
        "torch_upgrade_prompt": "Обновить PyTorch? [Y/n]: ",
        "torch_upgrade_yes": "Начинается обновление PyTorch...",
        "torch_upgrade_skip": "Обновление PyTorch пропущено",
        
        # Installation
        "install_title": "\nПроцесс установки",
        "install_selecting_wheel": "Выбор wheel для torch={torch}, CUDA={cuda}",
        "install_wheel_found": "✓ Найден: {wheel}",
        "install_wheel_not_found": "✗ Совместимый wheel не найден",
        "install_downloading": "Загрузка {file}...",
        "install_installing": "Установка SageAttention...",
        "install_success": "✓ Установка успешна",
        "install_failed": "✗ Установка не удалась",
        "install_torch_upgrading": "Обновление PyTorch до {version}...",
        
        # Rollback
        "rollback_title": "\nОпции отката",
        "rollback_prompt": "Установка не удалась. Откатить к предыдущей версии? [Y/n]: ",
        "rollback_starting": "Начинается откат...",
        "rollback_success": "✓ Откат выполнен успешно",
        "rollback_failed": "✗ Откат не удался",
        "rollback_skipped": "Откат пропущен",
        
        # Summary
        "summary_title": "\nИтоги установки",
        "summary_success": "✓ Установка завершена успешно",
        "summary_failed": "✗ Установка не удалась",
        "summary_previous_version": "Предыдущая версия: {version}",
        "summary_installed_version": "Установленная версия: {version}",
        "summary_torch_version": "PyTorch: {version}",
        "summary_cuda_version": "CUDA: {version}",
        "summary_errors": "Обнаружено ошибок: {count}",
        "summary_log_saved": "Детальный лог сохранен: {path}",
        
        # Errors
        "error_not_in_python_embeded": "ОШИБКА: Скрипт должен запускаться из папки python_embeded ComfyUI",
        "error_python_not_found": "ОШИБКА: Исполняемый файл Python не найден",
        "error_torch_not_installed": "ОШИБКА: PyTorch не установлен",
        "error_network": "ОШИБКА: Сбой сетевого подключения",
        "error_download_failed": "ОШИБКА: Не удалось загрузить {file}",
        "error_install_failed": "ОШИБКА: Установка не удалась - {reason}",
        
        # Cleanup
        "cleanup_title": "\nОчистка",
        "cleanup_removing": "Удаление временных файлов...",
        "cleanup_success": "✓ Очистка завершена",
        
        # Prompts
        "press_enter": "\nНажмите Enter для выхода...",
    }
}


def get_text(lang: str, key: str, **kwargs) -> str:
    """
    Get localized text by key with optional formatting.
    
    Args:
        lang: Language code ('en' or 'ru')
        key: Translation key
        **kwargs: Format arguments
    
    Returns:
        Formatted localized string
    """
    try:
        text = TRANSLATIONS[lang][key]
        return text.format(**kwargs) if kwargs else text
    except KeyError:
        # Fallback to English if key not found
        try:
            text = TRANSLATIONS["en"][key]
            return text.format(**kwargs) if kwargs else text
        except KeyError:
            return f"[MISSING: {key}]"


def get_system_language() -> str:
    """
    Detect system language and return appropriate code.
    
    Returns:
        'ru' for Russian systems, 'en' otherwise
    """
    import locale
    try:
        system_lang = locale.getdefaultlocale()[0]
        if system_lang and system_lang.startswith('ru'):
            return 'ru'
    except Exception:
        pass
    return 'en'
