"""
TRSA ComfyUI Installer - Localization Module
Version: 2.6.2
"""

import locale
from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        # WELCOME
        "welcome_title": "TRSA ComfyUI SageAttention Installer",
        "welcome_version": "Version: {version}",
        
        # LANGUAGE
        "lang_select_prompt": "Select language / Выберите язык:",
        "lang_option_en": "[1] English",
        "lang_option_ru": "[2] Русский",
        "lang_choice_prompt": "Your choice: ",
        
        # SYSTEM
        "check_title": "System Compatibility Check",
        "error_python_version": "ERROR: Python {version} is not supported. Minimum: 3.9",
        "error_torch_not_installed": "ERROR: PyTorch is not installed.",
        "check_sage_installed": "SageAttention: Installed (v{version})",
        "check_sage_not_installed": "SageAttention: Not installed",
        "check_compatible": "✓ System Compatible",
        "check_upgrade_needed": "⚠ Upgrade Recommended",
        "press_enter": "Press Enter to continue...",
        
        # TRITON
        "triton_title": "Triton Installation (Optional)",
        "triton_prompt": "Install Triton optimization? [Y/n]:",
        "triton_installing": "Installing Triton...",
        "triton_skipped": "Triton skipped.",
        
        # INSTALL
        "install_title": "SageAttention Installation",
        "install_wheel_found": "✓ Selected Wheel: {wheel}",
        "install_wheel_not_found": "✗ No compatible wheel found.",
        "install_downloading": "Downloading {file}...",
        "install_installing": "Installing SageAttention...",
        "error_download_failed": "ERROR: Failed to download {file}",
        
        # SUMMARY
        "cleanup_title": "Cleanup",
        "cleanup_success": "✓ Temporary files removed.",
        "summary_success": "✓ Installation Successful!",
        "summary_failed": "✗ Installation Failed.",
        "summary_next_steps": "Restart ComfyUI to apply changes.",
    },
    "ru": {
        # WELCOME
        "welcome_title": "TRSA ComfyUI SageAttention Установщик",
        "welcome_version": "Версия: {version}",
        
        # LANGUAGE
        "lang_select_prompt": "Выберите язык / Select language:",
        "lang_option_en": "[1] English",
        "lang_option_ru": "[2] Русский",
        "lang_choice_prompt": "Ваш выбор: ",
        
        # SYSTEM
        "check_title": "Проверка системы",
        "error_python_version": "ОШИБКА: Python {version} не поддерживается (минимум 3.9)",
        "error_torch_not_installed": "ОШИБКА: PyTorch не найден.",
        "check_sage_installed": "SageAttention: Установлен (v{version})",
        "check_sage_not_installed": "SageAttention: Не установлен",
        "check_compatible": "✓ Система совместима",
        "check_upgrade_needed": "⚠ Рекомендуется обновление",
        "press_enter": "Нажмите Enter для продолжения...",
        
        # TRITON
        "triton_title": "Установка Triton (Опционально)",
        "triton_prompt": "Установить Triton? [Y/n]:",
        "triton_installing": "Установка Triton...",
        "triton_skipped": "Triton пропущен.",
        
        # INSTALL
        "install_title": "Установка SageAttention",
        "install_wheel_found": "✓ Выбран файл: {wheel}",
        "install_wheel_not_found": "✗ Совместимый файл не найден.",
        "install_downloading": "Загрузка {file}...",
        "install_installing": "Установка SageAttention...",
        "error_download_failed": "ОШИБКА: Не удалось скачать {file}",
        
        # SUMMARY
        "cleanup_title": "Очистка",
        "cleanup_success": "✓ Временные файлы удалены.",
        "summary_success": "✓ Установка завершена успешно!",
        "summary_failed": "✗ Ошибка установки.",
        "summary_next_steps": "Перезапустите ComfyUI для применения изменений.",
    }
}

def get_text(lang: str, key: str, **kwargs: Any) -> str:
    """
    Retrieve translation safely. Falls back to English if key is missing.
    Ignores extra or missing format arguments to prevent crashes.
    """
    # 1. Try requested language, then fallback to 'en'
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    text = lang_dict.get(key)
    
    if not text:
        # Fallback to English if specific key missing in target lang
        text = TRANSLATIONS["en"].get(key, f"[MISSING: {key}]")

    # 2. Safe Formatting
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            # Return text as-is if formatting fails (e.g., missing placeholder in string)
            return text
        except Exception:
            return text
            
    return text

def get_system_language() -> str:
    try:
        sys_lang = locale.getdefaultlocale()[0]
        if sys_lang and sys_lang.startswith("ru"):
            return "ru"
    except Exception:
        pass
    return "en"
