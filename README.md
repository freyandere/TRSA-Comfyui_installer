# ComfyUI Accelerator — v2.5

Ускоряет ComfyUI в 2–3 раза на Windows за счёт Triton и SageAttention.

**Языки**: [English](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main) | [Русский](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main-ru)

## Содержание

- [Что нового в v2.5](#что-нового-в-v25)
- [Быстрый старт](#быстрый-старт)
- [Установка](#установка)
- [Системные требования](#системные-требования)
- [Производительность](#производительность)
- [Возможности](#возможности)
- [Диагностика](#диагностика)
- [Решение проблем](#решение-проблем)
- [Лицензия](#лицензия)
- [Благодарности](#благодарности)
- [Поддержите проект](#поддержите-проект)
- [Ссылки](#ссылки)
- [Связанные проекты](#связанные-проекты)


## Что нового в v2.5

- Однофайловый core-установщик (installer_core.py) с безопасными HTTPS-загрузками и защищённой распаковкой ZIP.
- Пошаговый TUI без прогресс-баров: понятные статусы и итоговая сводка.
- Жёсткие проверки совместимости:
    - Torch “2.7.1+cu128” корректно определяется как 2.7.1.
    - CUDA строго 12.8 для целевой конфигурации.
    - SageAttention wheel валидируется по win_amd64, cpXY/abi3 и Torch 2.7.1.
- Жёсткий пин: triton-windows<3.4.
- Итоговый отчёт по каждому компоненту (torch/include+libs/triton/sageattention).


## Быстрый старт

1) Поместите .bat в:
ComfyUI_windows_portable/
└─ python_embeded/
├─ python.exe
└─ TRSA_installer.bat



2) Запустите .bat (двойной клик) — он скачает installer_core.py и выполнит его встроенным Python.
3) Следуйте подсказкам:

- При несовпадении Torch/CUDA предложит переустановку Torch 2.7.1 (CUDA 12.8) (~2.5ГБ).
- Порядок шагов: include/libs → Triton (<3.4) → SageAttention wheel.
- В конце — сводный отчёт.


## Установка

Устанавливается:

- Triton for Windows: `triton-windows<3.4`
- SageAttention 2.2.x (CUDA 12.8 + Torch 2.7.1)
- include/ и libs/ для портативного Python
- Служебные зависимости (минимально необходимое)


## Системные требования

- Windows 10/11 x64
- Встроенный Python 3.11/3.12 (ComfyUI portable)
- NVIDIA GPU (CUDA)
- До ~2.5ГБ трафика и места при переустановке Torch


## Производительность

- Типичное ускорение: 2–3× при корректных версиях Torch/CUDA и установленном Triton/SageAttention.
- Детальные бенчмарки см. в документации SageAttention.


## Возможности

- Авто-определение языка (EN/RU) с принудительным выбором через переменные среды.
- Пошаговая TUI-установка без прогресс-баров.
- Подробный финальный отчёт по компонентам.
- Строгие проверки совместимости и безопасные загрузки.


## Диагностика

- Быстрая проверка наличия include/libs.
- Валидация версий Torch/CUDA и соответствия wheel SageAttention.
- Проверка успешной установки Triton и импортов модулей.


## Решение проблем

- Torch/CUDA mismatch: подтвердите переустановку Torch до 2.7.1 (CUDA 12.8).
- “not a supported wheel”: проверьте cp-тег (cp311/cp312) и платформу (win_amd64).
- Сетевые/SSL ошибки: проверьте прокси/AV; повторите попытку (HTTPS + верификация сертификатов).
- Нет PowerShell: используется urllib; PowerShell только как резерв.


### **Ручная установка**

Если автоматическая установка не удалась:

```bash
# 1. Установите Triton вручную
python -m pip install -U "triton-windows<3.4"

# 2. Скачайте wheel SageAttention
# Посетите: https://github.com/freyandere/TRSA-Comfyui_installer/releases

# 3. Установите из wheel
python -m pip install sageattention-2.2.0*.whl

# 4. Проверьте установку
python -c "import triton, sageattention; print('Успех!')"
```


### **Получение помощи**

- **Проблемы**: [GitHub Issues](https://github.com/freyandere/TRSA-Comfyui_installer/issues)
- **Обсуждения**: [GitHub Discussions](https://github.com/freyandere/TRSA-Comfyui_installer/discussions)
- **Документация**: [Wiki](https://github.com/freyandere/TRSA-Comfyui_installer/wiki)


### **Руководящие принципы участия**

1. **Форкните** репозиторий
2. **Создайте** ветку функций: `git checkout -b feature/amazing-feature`
3. **Тщательно тестируйте** на чистой установке ComfyUI
4. **Документируйте** изменения в коде и README
5. **Отправьте** pull request с подробным описанием

## Лицензия

Лицензия Apache 2.0 - см. файл [LICENSE](LICENSE) для деталей.

## Благодарности

Особая благодарность разработчикам проектов, которые сделали это возможным:

- **[Triton Windows](https://github.com/woct0rdho/triton-windows)** - Порт Triton для Windows от [@woct0rdho](https://github.com/woct0rdho) и сообщества
- **[SageAttention](https://github.com/thu-ml/SageAttention)** - Квантованное внимание от исследователей Tsinghua University
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** - Мощный узловой интерфейс от [@comfyanonymous](https://github.com/comfyanonymous)
- **[Telegram каналу - Psy Eyes](https://t.me/Psy_Eyes)** - за хайлайт репозитория и поддержку комьюнити.
- **[Telegram каналу - FRALID | НАСМОТРЕННОСТЬ](https://t.me/fralid95)** - за поддержку и напутствия.

Без их невероятной работы этот проект был бы невозможен!

## Поддержите проект

Если ComfyUI Accelerator помог ускорить ваши рабочие процессы:

- **Поставьте звёздочку** репозиторию
- **Сообщайте** о встреченных проблемах
- **Предлагайте** новые функции
- **Вносите** улучшения
- **Делитесь** с сообществом


## Ссылки

- **Главный репозиторий**: [Английская ветка](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main)
- **Русская версия**: [Русская ветка](https://github.com/freyandere/TRSA-Comfyui_installer/tree/main-ru)
- **Проблемы**: [Сообщить о проблемах](https://github.com/freyandere/TRSA-Comfyui_installer/issues)
- **Обсуждения**: [Форум сообщества](https://github.com/freyandere/TRSA-Comfyui_installer/discussions)
- **Wiki**: [Документация](https://github.com/freyandere/TRSA-Comfyui_installer/wiki)


### **Связанные проекты**

- **[Triton Windows](https://github.com/woct0rdho/triton-windows)** - GPU компилятор для ускорения
- **[SageAttention](https://github.com/thu-ml/SageAttention)** - Квантованные механизмы внимания
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** - Узловой интерфейс для ИИ

*Сделано с любовью для сообщества ComfyUI*
