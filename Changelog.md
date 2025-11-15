# TRSA-ComfyUI Installer – Changelog

All notable changes to this project will be documented in this file.  
This project follows a simplified Keep a Changelog style and semantic-ish versioning (MAJOR.MINOR.PATCH).

---

## [2.6.1] – 2025-11-08

### Fixed
- Corrected SageAttention wheel filenames to exactly match files stored in the `wheels/` directory, including Python 3.13 builds.[attached_file:1]
- Fixed URL handling for wheel downloads so that special characters (like `+` encoded as `%2B`) no longer break links.[attached_file:1]
- Improved rollback logic so that restoring the previous SageAttention version is more reliable and better reported in the summary.[attached_file:1]

### Changed
- Refined compatibility checks between Python, PyTorch, CUDA and SageAttention to avoid installing unsupported combinations.[attached_file:1]
- Updated log messages and error texts for clearer diagnostics when environment checks fail.[attached_file:1]

---

## [2.6.0] – 2025-11-08

### Added
- Structured installation flow in `installer_core.py`:
  welcome → system check → optional PyTorch upgrade → Triton installation → SageAttention wheel selection and installation → rollback → cleanup → final summary.[attached_file:1]
- Strong environment validation via `SystemInfo`:
  - Minimum Python 3.9.  
  - Detection of PyTorch and CUDA via `torch.__version__` and `nvcc --version` fallback.  
  - Compatibility matrix `SUPPORTED_CONFIGS` for `py39` and `py313` with explicit `torch_version`, `cuda_version`, wheel name and `torch_install_cmd` per configuration.[attached_file:1]
- A dedicated localization module `installer_core_lang.py` with English (`en`) and Russian (`ru`) strings for all stages: language selection, system check, upgrades, installation, rollback, summary and errors.[attached_file:1]
- A one-click batch entrypoint `TRSA_installer.bat` that:
  - Verifies it is run from the `python_embeded` folder of a ComfyUI portable build.  
  - Downloads the latest Python installer scripts from GitHub.  
  - Launches them with the embedded `python.exe` without requiring a system-wide Python.[attached_file:1]

### Changed
- Reworked the main installer script to use clear, staged logic instead of a monolithic flow, making it easier to debug and extend.[attached_file:1]
- Standardized logging to a timestamped file (`TRSA_install_HH.MM-DD.MM.YYYY.log`) with function names and log levels for easier issue reporting.[attached_file:1]
- Improved user-facing messages and prompts in both languages, including better explanations for PyTorch upgrades and next steps after a successful installation.[attached_file:1]

### Fixed
- Prevented installation from proceeding when PyTorch is missing or too old, with clear error messages and exit codes.[attached_file:1]
- Reduced the chance of broken installs by adding disk space checks (including a higher threshold before PyTorch upgrade).[attached_file:1]
- Ensured that failures during SageAttention installation no longer leave the environment in a half-broken state thanks to rollback prompts and actions.[attached_file:1]

---

## [Unreleased]

### Planned
- Anonymous, opt-out telemetry to understand which Python/PyTorch/CUDA combinations are most common and where installations fail (without collecting personal data).[attached_file:1]
- Better progress indicators (text-based progress bar) for large downloads and long `pip` operations.[attached_file:1]
- Optional “dry-run” mode that only checks compatibility and shows what would be installed, without making any changes.[attached_file:1]

---

## Русский / Russian Notes

### [2.6.1] – 2025-11-08

**Исправлено**
- Приведены названия wheel-файлов SageAttention к реальным именам в папке `wheels/`, включая сборки под Python 3.13.[attached_file:1]
- Исправлена обработка URL для скачивания wheel-файлов, чтобы символ `+` (кодированный как `%2B`) не ломал ссылки.[attached_file:1]
- Улучшен сценарий отката SageAttention: откат более предсказуем и корректно отражается в итоговом отчёте.[attached_file:1]

**Изменено**
- Уточнены проверки совместимости Python / PyTorch / CUDA / SageAttention, чтобы не устанавливать неподдерживаемые сочетания.[attached_file:1]
- Обновлены тексты ошибок и логов, чтобы было проще понимать, что именно пошло не так.[attached_file:1]

### [2.6.0] – 2025-11-08

**Добавлено**
- Ступенчатый сценарий работы `installer_core.py`:
  приветствие → проверка системы → (опциональное) обновление PyTorch → установка Triton → выбор и установка SageAttention → откат (если нужен) → очистка → финальный отчёт.[attached_file:1]
- Сильная валидация окружения через структуру `SystemInfo`:
  - Контроль минимальной версии Python (3.9+).  
  - Определение версий PyTorch и CUDA через `torch` + fallback на `nvcc`.  
  - Таблица `SUPPORTED_CONFIGS` с жёстко прописанными парами `torch+cuda` и соответствующими wheel-файлами для `py39` и `py313`.[attached_file:1]
- Отдельный модуль локализации `installer_core_lang.py` с полной поддержкой английского и русского интерфейса на всех этапах.[attached_file:1]
- One-click батник `TRSA_installer.bat`, который:
  - Проверяет корректность папки запуска (`python_embeded`).  
  - Скачивает свежие версии Python-скриптов с GitHub.  
  - Запускает установщик через встроенный `python.exe`, не требуя системного Python.[attached_file:1]

**Изменено**
- Основной код установщика переработан для понятной поэтапной логики, что сильно упрощает отладку и будущие доработки.[attached_file:1]
- Логирование стандартизировано: читаемый формат, отдельный файл с датой/временем и именами функций.[attached_file:1]
- Уточнены сообщения для пользователя (вопросы про обновление PyTorch, подсказки по дальнейшим действиям после установки и т.п.).[attached_file:1]

**Исправлено**
- Остановка установки при отсутствии или неподдерживаемой версии PyTorch с понятным сообщением и корректным выходом.[attached_file:1]
- Снижена вероятность “полубитой” установки за счёт проверки свободного места и сценария отката.[attached_file:1]
- Устранены ошибки из-за несоответствия имён wheel-файлов и URL в репозитории.[attached_file:1]
