"""
TRSA ComfyUI Installer - Localization Module
Version: 2.6.0
Supported languages: English (en), Russian (ru)

This module provides all user-facing strings for the installer
in multiple languages with automatic system language detection.
"""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, Any]] = {
    "en": {
        # ====================================================================
        # WELCOME SCREEN
        # ====================================================================
        "welcome_title": "TRSA ComfyUI SageAttention Installer",
        "welcome_version": "Version: {version}",
        "welcome_separator": "=" * 70,

        # ====================================================================
        # LANGUAGE SELECTION
        # ====================================================================
        "lang_select_prompt": "Select language / Выберите язык:",
        "lang_option_en": "[1] English",
        "lang_option_ru": "[2] Русский",
        "lang_default": "[Press Enter for system default]",
        "lang_choice_prompt": "Your choice: ",
        "lang_invalid": "Invalid choice. Using system default language.",
        "lang_selected": "Language selected: English",

        # ====================================================================
        # SYSTEM CHECK
        # ====================================================================
        "check_title": "\nSystem Compatibility Check",
        "check_python": "Python version: {version}",
        "check_torch": "PyTorch version: {version}",
        "check_cuda": "CUDA version: {version}",
        "check_sage_installed": "SageAttention: Installed (v{version})",
        "check_sage_not_installed": "SageAttention: Not installed",
        "check_compatible": "✓ Your system is fully compatible",
        "check_upgrade_needed": "⚠ Upgrade recommended for optimal performance",
        "check_current_config": "Current: PyTorch {torch}, CUDA {cuda}",
        "check_target_config": "Recommended: PyTorch {torch}, CUDA {cuda}",

        # ====================================================================
        # PYTORCH UPGRADE
        # ====================================================================
        "torch_upgrade_title": "\nPyTorch Upgrade Available",
        "torch_upgrade_msg": "Current version: PyTorch {current} + CUDA {cuda}",
        "torch_upgrade_recommend": "Recommended: PyTorch {target} + CUDA {cuda_target}",
        "torch_upgrade_benefits": "Benefits: Better performance, latest features, improved stability",
        "torch_upgrade_prompt": "Would you like to upgrade PyTorch? [Y/n]: ",
        "torch_upgrade_yes": "Starting PyTorch upgrade process...",
        "torch_upgrade_skip": "Skipping PyTorch upgrade. Continuing with current version.",
        "torch_upgrade_continue": "Continuing with current PyTorch version after upgrade failure.",
        "triton_title": "\nTriton Installation (Optional)",
        "triton_prompt": "Install Triton optimization? [Y/n]: ",
        "triton_installing": "Installing Triton...",
        "triton_success": "✓ Triton installed successfully",
        "triton_failed": "✗ Triton installation failed (non-critical)",
        "triton_skipped": "Triton installation skipped",
        "cleanup_removing_package": "Removing {package}...",
        "error_disk_space": "ERROR: Insufficient disk space ({free} MB free, {required} MB required)",
        "error_torch_disk_space": "ERROR: Not enough disk space for PyTorch upgrade (requires about 3 GB).",

        # ====================================================================
        # INSTALLATION
        # ====================================================================
        "install_title": "\nInstallation Process",
        "install_selecting_wheel": "Selecting wheel for PyTorch {torch}, CUDA {cuda}, Python {python}",
        "install_wheel_found": "✓ Found compatible wheel: {wheel}",
        "install_wheel_not_found": "✗ No compatible wheel found for your configuration",
        "install_downloading": "Downloading {file}...",
        "install_download_progress": "  Progress: {percent}%",
        "install_installing": "Installing SageAttention (this may take a minute)...",
        "install_success": "✓ SageAttention installed successfully!",
        "install_failed": "✗ Installation failed",
        "install_torch_upgrading": "Upgrading PyTorch to version {version}...",

        # ====================================================================
        # ROLLBACK
        # ====================================================================
        "rollback_title": "\nRollback Options",
        "rollback_prompt": "Installation failed. Restore previous version? [Y/n]: ",
        "rollback_starting": "Starting rollback to previous version...",
        "rollback_success": "✓ Successfully restored previous version",
        "rollback_failed": "✗ Rollback failed",
        "rollback_skipped": "Rollback skipped by user",

        # ====================================================================
        # SUMMARY
        # ====================================================================
        "summary_title": "\nInstallation Summary",
        "summary_success": "✓ Installation completed successfully!",
        "summary_failed": "✗ Installation encountered errors",
        "summary_previous_version": "Previous SageAttention: {version}",
        "summary_installed_version": "Installed SageAttention: {version}",
        "summary_torch_version": "PyTorch: {version}",
        "summary_cuda_version": "CUDA: {version}",
        "summary_python_version": "Python: {version}",
        "summary_errors": "Errors encountered: {count}",
        "summary_log_saved": "Detailed log saved: {path}",
        "summary_next_steps": "\nNext steps:",
        "summary_next_step_1": "1. Restart ComfyUI",
        "summary_next_step_2": "2. SageAttention will be automatically used",
        "summary_next_step_3": "3. Check the log file if you encounter issues",

        # ====================================================================
        # ERRORS
        # ====================================================================
        "error_not_in_python_embeded": "ERROR: This script must be run from ComfyUI's python_embeded folder",
        "error_python_not_found": "ERROR: Python executable not found",
        "error_python_version": "ERROR: Python {version} is not supported. Minimum: 3.9",
        "error_torch_not_installed": "ERROR: PyTorch is not installed. Please install ComfyUI properly.",
        "error_network": "ERROR: Network connection failed. Check your internet connection.",
        "error_download_failed": "ERROR: Failed to download {file}",
        "error_install_failed": "ERROR: Installation failed - {reason}",
        "error_permission": "ERROR: Permission denied. Try running as administrator.",

        # ====================================================================
        # CLEANUP
        # ====================================================================
        "cleanup_title": "\nCleanup",
        "cleanup_removing": "Removing temporary files...",
        "cleanup_success": "✓ Cleanup completed",

        # ====================================================================
        # PROMPTS
        # ====================================================================
        "press_enter": "\nPress Enter to exit...",
        "confirm_yes": "Y",
        "confirm_no": "n",
    },

    "ru": {
        # ====================================================================
        # WELCOME SCREEN
        # ====================================================================
        "welcome_title": "TRSA ComfyUI SageAttention Установщик",
        "welcome_version": "Версия: {version}",
        "welcome_separator": "=" * 70,

        # ====================================================================
        # LANGUAGE SELECTION
        # ====================================================================
        "lang_select_prompt": "Выберите язык / Select language:",
        "lang_option_en": "[1] English",
        "lang_option_ru": "[2] Русский",
        "lang_default": "[Нажмите Enter для системного языка]",
        "lang_choice_prompt": "Ваш выбор: ",
        "lang_invalid": "Неверный выбор. Используется системный язык.",
        "lang_selected": "Выбран язык: Русский",

        # ====================================================================
        # SYSTEM CHECK
        # ====================================================================
        "check_title": "\nПроверка совместимости системы",
        "check_python": "Версия Python: {version}",
        "check_torch": "Версия PyTorch: {version}",
        "check_cuda": "Версия CUDA: {version}",
        "check_sage_installed": "SageAttention: Установлен (v{version})",
        "check_sage_not_installed": "SageAttention: Не установлен",
        "check_compatible": "✓ Ваша система полностью совместима",
        "check_upgrade_needed": "⚠ Рекомендуется обновление для оптимальной производительности",
        "check_current_config": "Текущая: PyTorch {torch}, CUDA {cuda}",
        "check_target_config": "Рекомендуется: PyTorch {torch}, CUDA {cuda}",

        # ====================================================================
        # PYTORCH UPGRADE
        # ====================================================================
        "torch_upgrade_title": "\nДоступно обновление PyTorch",
        "torch_upgrade_msg": "Текущая версия: PyTorch {current} + CUDA {cuda}",
        "torch_upgrade_recommend": "Рекомендуется: PyTorch {target} + CUDA {cuda_target}",
        "torch_upgrade_benefits": "Преимущества: Лучшая производительность, новые функции, стабильность",
        "torch_upgrade_prompt": "Хотите обновить PyTorch? [Y/n]: ",
        "torch_upgrade_yes": "Начинается процесс обновления PyTorch...",
        "torch_upgrade_skip": "Обновление PyTorch пропущено. Продолжаем с текущей версией.",
        "torch_upgrade_continue": "Продолжаем с текущей версией PyTorch после неудачного обновления.",
        "triton_title": "\nУстановка Triton (Опционально)",
        "triton_prompt": "Установить оптимизацию Triton? [Y/n]: ",
        "triton_installing": "Установка Triton...",
        "triton_success": "✓ Triton успешно установлен",
        "triton_failed": "✗ Установка Triton не удалась (некритично)",
        "triton_skipped": "Установка Triton пропущена",
        "cleanup_removing_package": "Удаление {package}...",
        "error_disk_space": "ОШИБКА: Недостаточно места ({free} МБ свободно, требуется {required} МБ)",
        "error_torch_disk_space": "ОШИБКА: Недостаточно места для обновления PyTorch (требуется около 3 ГБ).",

        # ====================================================================
        # INSTALLATION
        # ====================================================================
        "install_title": "\nПроцесс установки",
        "install_selecting_wheel": "Выбор wheel для PyTorch {torch}, CUDA {cuda}, Python {python}",
        "install_wheel_found": "✓ Найден совместимый wheel: {wheel}",
        "install_wheel_not_found": "✗ Совместимый wheel не найден для вашей конфигурации",
        "install_downloading": "Загрузка {file}...",
        "install_download_progress": "  Прогресс: {percent}%",
        "install_installing": "Установка SageAttention (может занять минуту)...",
        "install_success": "✓ SageAttention успешно установлен!",
        "install_failed": "✗ Установка не удалась",
        "install_torch_upgrading": "Обновление PyTorch до версии {version}...",

        # ====================================================================
        # ROLLBACK
        # ====================================================================
        "rollback_title": "\nОпции отката",
        "rollback_prompt": "Установка не удалась. Восстановить предыдущую версию? [Y/n]: ",
        "rollback_starting": "Начинается откат к предыдущей версии...",
        "rollback_success": "✓ Предыдущая версия успешно восстановлена",
        "rollback_failed": "✗ Откат не удался",
        "rollback_skipped": "Откат пропущен пользователем",

        # ====================================================================
        # SUMMARY
        # ====================================================================
        "summary_title": "\nИтоги установки",
        "summary_success": "✓ Установка завершена успешно!",
        "summary_failed": "✗ При установке возникли ошибки",
        "summary_previous_version": "Предыдущая версия SageAttention: {version}",
        "summary_installed_version": "Установленная версия SageAttention: {version}",
        "summary_torch_version": "PyTorch: {version}",
        "summary_cuda_version": "CUDA: {version}",
        "summary_python_version": "Python: {version}",
        "summary_errors": "Обнаружено ошибок: {count}",
        "summary_log_saved": "Детальный лог сохранён: {path}",
        "summary_next_steps": "\nСледующие шаги:",
        "summary_next_step_1": "1. Перезапустите ComfyUI",
        "summary_next_step_2": "2. SageAttention будет использоваться автоматически",
        "summary_next_step_3": "3. Проверьте лог-файл при возникновении проблем",

        # ====================================================================
        # ERRORS
        # ====================================================================
        "error_not_in_python_embeded": "ОШИБКА: Скрипт должен запускаться из папки python_embeded ComfyUI",
        "error_python_not_found": "ОШИБКА: Исполняемый файл Python не найден",
        "error_python_version": "ОШИБКА: Python {version} не поддерживается. Минимум: 3.9",
        "error_torch_not_installed": "ОШИБКА: PyTorch не установлен. Установите ComfyUI правильно.",
        "error_network": "ОШИБКА: Сбой подключения к сети. Проверьте интернет-соединение.",
        "error_download_failed": "ОШИБКА: Не удалось загрузить {file}",
        "error_install_failed": "ОШИБКА: Установка не удалась - {reason}",
        "error_permission": "ОШИБКА: Доступ запрещён. Попробуйте запустить от имени администратора.",

        # ====================================================================
        # CLEANUP
        # ====================================================================
        "cleanup_title": "\nОчистка",
        "cleanup_removing": "Удаление временных файлов...",
        "cleanup_success": "✓ Очистка завершена",

        # ====================================================================
        # PROMPTS
        # ====================================================================
        "press_enter": "\nНажмите Enter для выхода...",
        "confirm_yes": "Y",
        "confirm_no": "n",
    },
}


def get_text(lang: str, key: str, **kwargs: Any) -> str:
    try:
        text = TRANSLATIONS[lang][key]
        return text.format(**kwargs) if kwargs else text
    except KeyError:
        try:
            text = TRANSLATIONS["en"][key]
            return text.format(**kwargs) if kwargs else text
        except KeyError:
            return f"[MISSING TRANSLATION: {key}]"
    except Exception as e:
        return f"[TRANSLATION ERROR: {key} - {str(e)}]"


def get_system_language() -> str:
    import locale

    try:
        system_lang, _ = locale.getdefaultlocale()
        if system_lang and system_lang.startswith("ru"):
            return "ru"
    except Exception:
        pass
    return "en"


def get_available_languages() -> list:
    return list(TRANSLATIONS.keys())
