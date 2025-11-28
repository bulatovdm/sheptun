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
# Модель Whisper
SHEPTUN_MODEL=medium              # tiny, base, small, medium, large
SHEPTUN_DEVICE=                   # cpu, cuda, mps (auto если пусто)

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

# Фильтрация галлюцинаций Whisper
SHEPTUN_HALLUCINATIONS=Продолжение следует...,Спасибо за просмотр!

# Метод ввода текста
SHEPTUN_USE_CLIPBOARD=false       # true — через буфер обмена, false — через CGEvent
SHEPTUN_KEY_DELAY=0.02            # Задержка между событиями клавиатуры (сек)

# Производительность
SHEPTUN_WARMUP_INTERVAL=120       # Интервал прогрева модели (сек), 0 — отключить
```

## Фильтрация галлюцинаций

Whisper иногда генерирует ложные транскрипции на тишине или шуме — так называемые "галлюцинации". Типичные примеры: "Продолжение следует...", "Спасибо за просмотр!", "Субтитры сделал...".

Sheptun автоматически фильтрует известные галлюцинации. Если распознанный текст совпадает с фразой из списка, он игнорируется.

Можно добавить свои фразы через переменную `SHEPTUN_HALLUCINATIONS` (через запятую).

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
