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
```

### Menubar приложение (macOS)

```bash
sheptun install-app         # Установить приложение в /Applications
sheptun restart             # Перезапустить приложение
```

После установки Sheptun появится в menubar. Приложение работает в фоне без терминала.

## Конфигурация

Создайте файл `.env` в корне проекта или настройте переменные окружения:

```bash
SHEPTUN_MODEL=medium              # Модель: tiny, base, small, medium, large
SHEPTUN_DEVICE=                   # Устройство: cpu, cuda, mps (auto если пусто)
SHEPTUN_SILENCE_DURATION=0.3      # Пауза для определения конца фразы (сек)
SHEPTUN_DEBUG=false               # Включить логирование
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
