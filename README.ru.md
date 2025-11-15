<div align="center">

# ⚡ TRSA ComfyUI Installer

Ускоритель ComfyUI для Windows на базе Triton и SageAttention.

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/releases)
[![GitHub stars](https://img.shields.io/github/stars/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Windows-brightgreen.svg)](https://github.com/freyandere/TRSA-Comfyui_installer)
[![Python](https://img.shields.io/badge/Python-3.9%20%E2%80%93%203.13-blue)](https://python.org)

[![starline](https://starlines.qoo.monster/assets/freyandere/TRSA-Comfyui_installer)](https://github.com/freyandere/TRSA-Comfyui_installer/stargazers)

Ускорение инференса до 30% на поддерживаемых Windows‑системах.

[English version](README.md)

[Быстрый старт](#быстрый-старт) • [Возможности](#возможности) • [Детали-установки](#детали-установки) • [Требования-и-производительность](#требования-и-производительность) • [Решение-проблем](#решение-проблем) • [Поддержать-проект](#поддержать-проект)

</div>

---

## Что такое TRSA?

TRSA (Triton + SageAttention) — это однокнопочный установщик, который ускоряет ComfyUI на Windows за счёт интеграции Triton и оптимизированных attention‑ядер SageAttention.  
Проект рассчитан на пользователей ComfyUI portable: художников, разработчиков и энтузиастов, которым важна скорость генерации.

### Ключевые преимущества

- Ускорение инференса до 30% на поддерживаемых GPU.  
- Установка в один клик через `TRSA_installer.bat`.  
- Безопасность:
  - проверка окружения;
  - валидация версий;
  - возможность отката изменений.  
- Два языка интерфейса: русский и английский.  
- Поддержка portable‑сборок ComfyUI (встроенный Python, без установки Python в систему).

---

## Быстрый старт

> Требования:  
> - Windows 10/11 x64  
> - Видеокарта NVIDIA (серия RTX 20/30/40, поддержка CUDA)  
> - Portable‑сборка ComfyUI со встроенным Python (3.9–3.13)

### Метод 1: Автоматическая установка (рекомендуется)

1. Поместите установщик в папку с ComfyUI.

ComfyUI_windows_portable/
└─ python_embeded/
├─ python.exe
└─ TRSA_installer.bat ← сюда

text

2. Скачайте и запустите.

- Скачайте `TRSA_installer.bat` со страницы Releases.  
- Запустите `TRSA_installer.bat` двойным кликом.  
- Следуйте подсказкам:
  - выберите язык (EN / RU);
  - дождитесь проверки Python, PyTorch, CUDA и SageAttention;
  - при необходимости согласитесь на обновление PyTorch до поддерживаемой версии.

3. Перезапустите ComfyUI.

- Запустите привычный `run.bat` или другой стартовый скрипт ComfyUI.  
- В консоли вы должны увидеть строки вида:
  - `pytorch version: 2.9.0+cu130` (или другую поддерживаемую связку);  
  - `Using sage attention`.  

Если эти строки есть и ComfyUI запускается без ошибок, TRSA установлен и активен.

---

## Возможности

### Технические особенности

- Интеграция SageAttention 2.2.x:
- оптимизированные attention‑ядра для ускорения и снижения потребления памяти;
- готовые wheel‑файлы для:
 - Python 3.9 + CUDA 12.4–13.0 (несколько версий Torch);
 - Python 3.13 + CUDA 13.0 (Torch 2.9.0 и 2.10.0).  
- Поддержка Triton на Windows:
- для Python 3.13 — установка через `triton-windows` с PyPI;  
- для Python 3.9–3.12 — опциональная установка из готовых wheel‑файлов (где доступны).  
- Версионно‑осознанный инсталлятор:
- проверяет версии Python, PyTorch, CUDA и SageAttention;
- использует таблицу совместимости `SUPPORTED_CONFIGS`, чтобы подбирать только поддерживаемые сочетания.

### Особенности установки

- Умное определение окружения:
- автоматическое определение языка (RU/EN);
- чтение версий Python, PyTorch и CUDA из встроенного окружения ComfyUI.  
- Проверки совместимости:
- установка продолжается только при подходящих версиях;
- если версия PyTorch слишком старая или “неудачная”, предлагается обновление до рекомендованной.  
- Откат (rollback):
- если SageAttention уже был установлен, перед установкой сохраняется текущая версия;
- при неудаче установки можно откатиться обратно.  
- Очистка:
- временные файлы (скачанные wheel‑ы, Triton‑wheel) удаляются после работы скрипта.  
- Логирование:
- создаётся подробный лог в текущей директории:
 - `TRSA_install_ЧЧ.ММ-ДД.ММ.ГГГГ.log`.

### Пользовательский опыт

- Пошаговый консольный интерфейс:
- понятные вопросы и статусы;
- все сообщения идут через модуль локализации.  
- Минимальные зависимости:
- используется только встроенный Python из portable‑сборки ComfyUI;
- не требуется отдельный Python в системе.  
- Прозрачный финальный отчёт:
- предыдущая и новая версия SageAttention;
- версии Python, PyTorch, CUDA;
- список ошибок (если были);
- путь к лог‑файлу.

---

## Детали установки

### Что делает установщик

В зависимости от вашего окружения, скрипт:

- Проверяет текущее состояние:
- версию Python (`sys.version_info`);
- версию PyTorch и CUDA (через `torch.__version__` и, при наличии, `nvcc --version`);
- наличие и версию SageAttention.  
- При необходимости предлагает обновить PyTorch:
- например, до `torch==2.9.0+cu130` на CUDA 13.0 для современных RTX‑карт.  
- Устанавливает Triton:
- для Python 3.13 — командой:
 ```
 python -m pip install -U "triton-windows<3.6"
 ```
- для других поддерживаемых версий Python — скачивает готовый wheel из репозитория Triton‑Windows и устанавливает его.  
- Устанавливает SageAttention:
- подбирает подходящий wheel по связке Python + PyTorch + CUDA;
- скачивает его из папки `wheels/` этого репозитория;
- устанавливает/переустанавливает через `pip`.

### Поддерживаемые сочетания версий

Инсталлятор рассчитан на следующие диапазоны:

- Python:
- 3.9 (классические portable‑сборки ComfyUI);  
- 3.13 (новые portable‑сборки с экспериментальным Python 3.13).  
- PyTorch:
- 2.5.1+ на CUDA 12.4;  
- 2.6.0+ на CUDA 12.6;  
- 2.7.1 / 2.8.0 на CUDA 12.8;  
- 2.9.0 / 2.10.0 на CUDA 13.0.  
- CUDA:
- 12.4–13.0 на видеокартах NVIDIA RTX.

Если ваша текущая связка выходит за эти границы, установщик либо предложит обновить PyTorch, либо откажется устанавливать неподдерживаемые wheel‑ы.

---

## Требования и производительность

<details>
<summary><strong>Практическая совместимость GPU</strong></summary>

| GPU | Статус | Комментарии |
|-----|--------|-------------|
| RTX 4090 | ✅ Полная поддержка | Оптимальная производительность, большой запас VRAM |
| RTX 4080 | ✅ Высокая производительность | ~30% ускорение относительно базового состояния |
| RTX 3090 | ✅ Полная поддержка SageAttention | Хорош для тяжёлых моделей |
| RTX 3080 | ✅ Хорошая производительность | Комфортный запуск SDXL |
| RTX 3070 | ✅ Поддерживается | 8 GB VRAM на грани, но пригодно |
| RTX 3060 (12 GB) | ✅ Поддерживается | Хороший середнячок |
| RTX 20xx / GTX 10xx | ⚠️ Ограниченная поддержка | Ускорение меньше, возможны ограничения по VRAM |
| Non‑NVIDIA / iGPU | ❌ Не поддерживается | Нужны CUDA и Tensor Cores |

</details>

---

## Решение проблем

<details>
<summary><strong>“Torch/CUDA version mismatch”</strong></summary>

Инсталлятор обнаружил, что ваша текущая связка PyTorch/CUDA не поддерживается доступными wheel‑ами SageAttention.

**Что делать:**

- Разрешить инсталлятору обновить PyTorch до рекомендуемой версии (например, `2.9.0+cu130` для CUDA 13.0).  
- Либо обновить PyTorch вручную:

python -m pip install "torch==2.9.0+cu130" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

text

После этого повторно запустить `TRSA_installer.bat`.
</details>

<details>
<summary><strong>“Not a supported wheel”</strong></summary>

Обычно значит, что:

- версия Python не совпадает с поддерживаемой (wheel не для cp39/abi3 или cp313);  
- архитектура не 64‑битная;  
- wheel не соответствует вашей связке CUDA/PyTorch.

**Проверьте:**

- что вы используете portable‑сборку ComfyUI со встроенным Python 3.9–3.13;  
- что PyTorch либо уже в поддерживаемом диапазоне, либо вы согласились на обновление.
</details>

<details>
<summary><strong>“Network/SSL errors”</strong></summary>

**Проверьте:**

- интернет‑соединение;  
- настройки firewall / антивируса (для `python.exe` и утилит загрузки);  
- попробуйте запустить `TRSA_installer.bat` от имени администратора.

Если автоматический режим постоянно падает на сети, можно:

1. Вручную скачать подходящий wheel SageAttention из папки `wheels/` этого репозитория.  
2. Выполнить из `python_embeded`:

python -m pip install <имя_файла_wheel>.whl

text
</details>

<details>
<summary><strong>“Failed to find cuobjdump.exe / nvdisasm.exe”</strong></summary>

Это предупреждения от Triton: не найдены утилиты CUDA Toolkit (`cuobjdump`/`nvdisasm`).  
Они **не мешают** работе SageAttention и в большинстве случаев могут быть проигнорированы для обычного использования ComfyUI.
</details>

---

## Поддержка и обратная связь

- О проблемах сообщайте в Issues этого репозитория.  
- Идеи и предложения — в Discussions.  
- Для ускорения диагностики прикладывайте:
- свежий `TRSA_install_*.log`;  
- фрагмент лога запуска ComfyUI (строки с PyTorch / CUDA / SageAttention).

---

## Вклад в проект

1. Сделайте форк репозитория.  
2. Создайте ветку для своей фичи:

git checkout -b feature/my-improvement

text

3. Протестируйте изменения на чистой portable‑сборке ComfyUI.  
4. Обновите документацию при необходимости (`README.md`, `README.ru.md`, `CHANGELOG.md`).  
5. Откройте pull request с понятным описанием и шагами для воспроизведения.

---

## Благодарности

Проект опирается на работу:

- Triton Windows — порт Triton под Windows от @woct0rdho.  
- SageAttention — квантованные attention‑ядра от команды Tsinghua University.  
- ComfyUI — нодовая UI‑оболочка от @comfyanonymous и сообщества.

---

## Лицензия

Проект распространяется по лицензии Apache 2.0. Подробности — в файле `LICENSE`.

---

## Поддержать проект

Если TRSA ускорил ваши пайплайны в ComfyUI:

- Поставьте звёздочку репозиторию.  
- Сообщайте о багах.  
- Предлагайте улучшения.  
- Делитесь ссылкой с другими пользователями ComfyUI.  
- Присылайте pull request‑ы с улучшениями.
