# Sheptun

Голосовое управление терминалом на русском языке.

## Установка

```bash
pip install -e .

# Активировать виртуальное окружение
source .venv/bin/activate
```

## Использование

### CLI

```bash
sheptun listen              # Запуск в терминале
sheptun listen -m medium    # С указанной моделью Whisper
sheptun listen --debug      # С логированием
sheptun test-mic            # Проверка микрофона
sheptun list-commands       # Список команд
sheptun list-models         # Показать загруженные модели
sheptun cleanup-models      # Удалить неиспользуемые модели
sheptun clear-dataset       # Очистить датасет для fine-tuning
```

### Menubar приложение (macOS)

```bash
sheptun install-app         # Установить приложение в /Applications
sheptun restart             # Перезапустить приложение
```

После установки Sheptun появится в menubar. Приложение работает в фоне без терминала.

### Горячие клавиши

- **Toggle** (`Ctrl+Option+S`) — включить/выключить запись
- **Push-to-talk** (`Ctrl+Option+Space`) — удерживать для записи

### Быстрый запуск

Скрипт `run.sh` позволяет запустить sheptun из корня проекта:

```bash
./run.sh                    # Показать справку
./run.sh listen             # Запуск CLI
./run.sh install-app        # Установить menubar приложение

# Разработка
./run.sh check              # Запустить все проверки (ruff, mypy, pyright)
./run.sh test               # Запустить тесты
./run.sh format             # Отформатировать код
./run.sh coverage           # Тесты с покрытием
```

## Конфигурация

Создайте файл `.env` в корне проекта или настройте переменные окружения:

```bash
# Распознавание речи
SHEPTUN_RECOGNIZER=whisper        # whisper или apple
SHEPTUN_MODEL=medium              # tiny, base, small, medium, large (только для whisper)
SHEPTUN_DEVICE=                   # cpu, cuda, mps (auto если пусто, только для whisper)

# Voice Activity Detection
SHEPTUN_VAD_TYPE=energy           # energy или silero
SHEPTUN_ENERGY_THRESHOLD=0.01     # Порог энергии для energy VAD
SHEPTUN_SILENCE_DURATION=0.5      # Пауза для определения конца фразы (сек)
SHEPTUN_MIN_SPEECH_DURATION=0.2   # Минимальная длительность речи (сек)
SHEPTUN_MAX_SPEECH_DURATION=30.0  # Максимальная длительность речи (сек)

# Горячие клавиши (menubar)
SHEPTUN_HOTKEY_TOGGLE=<ctrl>+<alt>+s       # Toggle режим
SHEPTUN_HOTKEY_PTT=<ctrl>+<alt>+<space>    # Push-to-talk режим

# Отладка
SHEPTUN_DEBUG=false               # Включить логирование
SHEPTUN_LOG_FILE=logs/sheptun.log # Путь к файлу логов

# Приложение
SHEPTUN_APP_PATH=/Applications/Sheptun.app

# Сбор данных для fine-tuning
SHEPTUN_RECORD_DATASET=false      # Записывать аудио и транскрипции
SHEPTUN_DATASET_PATH=dataset      # Относительно директории запуска

# Автоматические пробелы
SHEPTUN_AUTO_SPACE=true           # Добавлять пробел перед текстом

# Коррекция орфографии (экспериментально)
SHEPTUN_SPELL_CORRECTION=none     # none или t5-russian

# Фильтрация галлюцинаций Whisper
SHEPTUN_HALLUCINATIONS=Продолжение следует...,Спасибо за просмотр!

# Метод ввода текста
SHEPTUN_USE_CLIPBOARD=false       # true — через буфер обмена, false — через CGEvent
SHEPTUN_KEY_DELAY=0.02            # Задержка между событиями клавиатуры (сек)

# Производительность
SHEPTUN_WARMUP_INTERVAL=120       # Интервал прогрева модели (сек), 0 — отключить
```

## Движки распознавания

Sheptun поддерживает два движка распознавания речи:

### Whisper (по умолчанию)

Локальная модель от OpenAI. Работает оффлайн, высокое качество для русского языка.

```bash
SHEPTUN_RECOGNIZER=whisper
SHEPTUN_MODEL=medium
```

Модели: `tiny`, `base`, `small`, `medium`, `large`

### Apple Speech Framework

Нативная система распознавания macOS. Быстрее запускается, использует меньше памяти.

```bash
SHEPTUN_RECOGNIZER=apple
SHEPTUN_APPLE_LOCALE=ru-RU  # или en-US, en-GB, de-DE и т.д.
```

**Особенности:**
- Использует встроенный Speech Framework (SFSpeechRecognizer)
- Работает в on-device режиме (без интернета)
- Не требует загрузки моделей Whisper
- Качество распознавания может быть выше для русского языка
- Поддержка множества языков (настраивается через `SHEPTUN_APPLE_LOCALE`)
- Только для macOS

**Ограничения:**
- Требует системного разрешения на распознавание речи
- Поддержка языков зависит от версии macOS
- Не определяет язык автоматически, нужно выбрать один

## Фильтрация галлюцинаций

Whisper иногда генерирует ложные транскрипции на тишине или шуме — так называемые "галлюцинации". Типичные примеры: "Продолжение следует...", "Спасибо за просмотр!", "Субтитры сделал...".

Sheptun автоматически фильтрует известные галлюцинации. Если распознанный текст совпадает с фразой из списка, он игнорируется.

Можно добавить свои фразы через переменную `SHEPTUN_HALLUCINATIONS` (через запятую).

Примечание: фильтрация галлюцинаций работает только для Whisper. Apple Speech не генерирует такие галлюцинации.

## Голосовые команды

- `клод` → вводит `claude`
- `таб`, `энтер`, `эскейп`, `пробел` → клавиши
- `вверх`, `вниз`, `влево`, `вправо` → стрелки
- `удали`, `клир` → backspace, очистить строку
- `слэш хелп` → `/help` + Enter
- `скажи <текст>` → вводит текст
- `команды` → показать справку
- `стоп` → остановить прослушивание

## Настройка команд

Создайте файл `~/.config/sheptun/commands.yaml` или `./sheptun.yaml`:

```yaml
control_commands:
  "моя команда": { type: "key", value: "f5" }

slash_commands:
  "мой слэш": "/my-command"
```

## Методы ввода текста

Sheptun поддерживает два метода ввода текста:

### Clipboard (по умолчанию)

Текст вставляется через буфер обмена (Cmd+V). Перед вставкой сохраняет содержимое буфера и восстанавливает после.

**Ограничения:**
- Небольшая задержка (~50ms) на вставку
- Если быстро нажать Cmd+V сразу после голосового ввода, может вставиться распознанный текст вместо оригинала

### CGEvent (по умолчанию)

Текст вводится напрямую через macOS Quartz Events (`CGEventKeyboardSetUnicodeString`).

**Ограничения:**
- При слишком низком `SHEPTUN_KEY_DELAY` возможно дублирование текста
- Рекомендуемое значение: 0.015-0.02 секунд

Переключение через переменную `SHEPTUN_USE_CLIPBOARD` (по умолчанию `false`).

## Автоматические пробелы

При последовательном вводе текста Sheptun автоматически добавляет пробел перед словами, чтобы они не склеивались.

Логика работы:
- Если курсор находится после текста (позиция > 0) — добавляет пробел
- Если поле пустое (позиция = 0) — пробел не добавляется
- Состояние отслеживается отдельно для каждого окна

**Ограничения:**
- В VS Code и других Electron-приложениях macOS Accessibility API не работает
- Для таких приложений используется fallback: отслеживание по окнам
- После отправки сообщения в чате пробел может добавиться к новому сообщению

Отключить: `SHEPTUN_AUTO_SPACE=false`

## Коррекция орфографии

Экспериментальная функция для исправления ошибок распознавания с помощью T5-russian модели.

```bash
# Включить в .env
SHEPTUN_SPELL_CORRECTION=t5-russian

# Загрузить модель
sheptun download-spelling
```

По умолчанию отключено (`none`). Модель занимает ~800MB и требует дополнительных зависимостей.
