# Sheptun

Голосовое управление терминалом на русском языке.

## Установка

```bash
pip install -e .
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
```

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
