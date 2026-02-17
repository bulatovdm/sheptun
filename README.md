# Sheptun

Голосовое управление терминалом на русском языке.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate

# Настроить конфигурацию перед установкой
cp .env.example .env
# Отредактируйте .env — выберите SHEPTUN_RECOGNIZER (whisper, mlx или apple)

# Установить базовые зависимости
pip install -e .

# Для MLX Whisper (рекомендуется для Apple Silicon)
pip install -e ".[mlx]"

# Для разработки
pip install -e ".[dev]"
```

**Важно:** настройте `.env` до сборки приложения (`sheptun install-app`), так как от выбранного движка (`SHEPTUN_RECOGNIZER`) зависит, какие зависимости нужно установить. Например, для `mlx` необходим `pip install -e ".[mlx]"`.

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

# Remote-ввод (Universal Control)
sheptun serve               # Запустить remote-сервер (приёмник)
sheptun remote-test         # Проверить подключение к remote

# Верификация транскрипций
sheptun verify-dataset      # Верифицировать транскрипции через Claude
sheptun verify-dataset -n 10  # Тест на 10 записях
sheptun verify-dataset --retry  # Повторить ошибочные
sheptun verify-dataset --reset  # Сбросить и обработать заново
sheptun verify-status       # Статус верификации
sheptun verify-export       # Экспорт в JSONL
```

### Menubar приложение (macOS)

```bash
sheptun install-app         # Установить приложение в /Applications
sheptun restart             # Перезапустить приложение
sheptun enable-autostart    # Включить автозапуск при старте системы
sheptun disable-autostart   # Отключить автозапуск
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
SHEPTUN_RECOGNIZER=whisper        # whisper, mlx или apple
SHEPTUN_MODEL=medium              # tiny, base, small, medium, large, turbo
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

# Remote-ввод (Universal Control)
SHEPTUN_REMOTE_ENABLED=false      # Включить отправку текста на remote
SHEPTUN_REMOTE_SERVE=false        # Принимать текст от remote (режим сервера)
SHEPTUN_REMOTE_HOST=              # Хост remote-сервера (пусто — Bonjour)
SHEPTUN_REMOTE_PORT=7849          # Порт remote-сервера
SHEPTUN_REMOTE_TOKEN=             # Shared secret для авторизации
SHEPTUN_REMOTE_AUTO_DETECT=true   # Автоопределение по позиции курсора
```

## Движки распознавания

Sheptun поддерживает три движка распознавания речи:

### Whisper (по умолчанию)

Локальная модель от OpenAI. Работает оффлайн, высокое качество для русского языка.

```bash
SHEPTUN_RECOGNIZER=whisper
SHEPTUN_MODEL=medium
```

Модели: `tiny`, `base`, `small`, `medium`, `large`, `turbo`

### MLX Whisper (рекомендуется для Apple Silicon)

Нативная GPU-акселерация на Apple Silicon через Metal. В ~5-6x быстрее стандартного Whisper (~0.4 сек вместо ~2.5 сек). Требует macOS 13.5+.

```bash
pip install -e ".[mlx]"

SHEPTUN_RECOGNIZER=mlx
SHEPTUN_MODEL=turbo
```

Модели: `tiny`, `base`, `small`, `medium`, `large`, `turbo`

При первом запуске модель скачивается автоматически из HuggingFace (~1.6 ГБ для turbo). Прогресс скачивания отображается в menubar.

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

## Верификация транскрипций

Инструмент для проверки и исправления транскрипций Whisper через Claude. Использует Claude Agent SDK с подпиской Max (без API ключа, $0).

```bash
# Установить зависимости
pip install -e ".[verification]"

# Тестовый прогон на 10 записях
sheptun verify-dataset --limit 10

# Полный прогон всего датасета
sheptun verify-dataset

# Повторить только ошибочные записи
sheptun verify-dataset --retry

# Сбросить все результаты и обработать заново
sheptun verify-dataset --reset

# Посмотреть статистику
sheptun verify-status

# Экспортировать результаты в JSONL
sheptun verify-export -o dataset/transcripts_verified.jsonl
```

Настройка в `.env`:

```bash
# Модель Claude для верификации (по умолчанию — модель Agent SDK)
SHEPTUN_VERIFY_MODEL=claude-haiku-4-5-20251001
```

Результаты хранятся в SQLite (`dataset/verification.db`) — поддерживает инкрементальную обработку и возобновление после прерывания.

## Remote-ввод (Universal Control)

Sheptun поддерживает передачу текста между Mac-ами. Это полезно при использовании Universal Control: Mac Studio распознаёт речь и отправляет текст на MacBook, где курсор.

### Как это работает

1. **Mac Studio** (распознавание) — определяет позицию курсора. Если курсор за пределами локальных экранов (ушёл на MacBook через Universal Control), текст отправляется по HTTP на MacBook.
2. **MacBook** (приёмник) — запускает `sheptun serve`, принимает текст и вводит его через CGEvent/clipboard как обычно.
3. **Обнаружение** — MacBook рекламирует себя через Bonjour (`_sheptun._tcp`), Mac Studio находит его автоматически.
4. **Конфликты** — если на MacBook идёт локальная запись (PTT/Toggle), сервер отклоняет входящий текст (HTTP 409), и Mac Studio вводит его локально.

### Настройка

**Mac Studio** (отправитель) — `.env`:

```bash
SHEPTUN_REMOTE_ENABLED=true          # Включить отправку на remote
SHEPTUN_REMOTE_TOKEN=my-secret       # Общий токен авторизации
SHEPTUN_REMOTE_AUTO_DETECT=true      # Автоопределение по позиции курсора
# SHEPTUN_REMOTE_HOST=              # Явный хост (если без Bonjour)
# SHEPTUN_REMOTE_PORT=7849          # Порт (по умолчанию 7849)
```

**MacBook** (приёмник) — `.env`:

```bash
SHEPTUN_REMOTE_SERVE=true            # Принимать текст от remote
SHEPTUN_REMOTE_TOKEN=my-secret       # Тот же токен
```

### CLI-команды

```bash
# На MacBook — запустить сервер в foreground (для отладки)
sheptun serve
sheptun serve --port 7849 --token my-secret

# На Mac Studio — проверить подключение
sheptun remote-test                      # Bonjour-обнаружение
sheptun remote-test macbook.local        # По имени хоста
sheptun remote-test 192.168.1.42         # По IP
```

### Сценарии

| Ситуация | Поведение |
|----------|-----------|
| Курсор на Mac Studio | Текст вводится локально |
| Курсор на MacBook (UC) | Текст отправляется на MacBook |
| MacBook записывает голос | Remote отклонён → текст вводится локально |
| MacBook ушёл из сети | Bonjour-сервис пропал → текст вводится локально |
| MacBook отдельно (без UC) | Оба Sheptun работают независимо |

## Fine-tuning Whisper

Дообучение модели Whisper на верифицированных данных для улучшения распознавания русской речи в контексте управления терминалом. Поддерживает два метода: LoRA (рекомендуется) и полный fine-tuning. Оптимизирован для Apple Silicon (MPS).

```bash
# Установить зависимости (+ FFmpeg через brew install ffmpeg)
pip install -e ".[finetune]"

# Подготовить датасет из verification.db
sheptun finetune-prepare

# Запустить обучение (LoRA по умолчанию)
sheptun finetune-train

# С настройками
sheptun finetune-train --method lora --steps 4000 --batch-size 4

# Полный fine-tuning (больше памяти, выше качество)
sheptun finetune-train --method full

# Продолжить обучение с последнего checkpoint
sheptun finetune-train --resume

# Оценить модель (WER/CER: base vs fine-tuned)
sheptun finetune-eval

# Использовать fine-tuned модель
SHEPTUN_MODEL=models/whisper-sheptun sheptun listen
```

Настройка в `.env`:

```bash
SHEPTUN_FINETUNE_MODEL=large             # Базовая модель (tiny/base/small/medium/large)
SHEPTUN_FINETUNE_METHOD=lora             # Метод обучения (lora, full)
SHEPTUN_FINETUNE_STEPS=4000              # Количество шагов
SHEPTUN_FINETUNE_BATCH_SIZE=4            # Размер батча
SHEPTUN_FINETUNE_LR=1e-5                 # Learning rate
SHEPTUN_FINETUNE_OUTPUT=models/whisper-sheptun  # Куда сохранять
SHEPTUN_FINETUNE_MIN_CONFIDENCE=medium   # Минимальный confidence из verification.db
```

Подробнее: [docs/finetune-spec.md](docs/finetune-spec.md)
