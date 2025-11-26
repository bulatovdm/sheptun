# Sheptun

Голосовое управление терминалом на русском языке.

## Установка

```bash
pip install -e .
```

## Использование

```bash
sheptun listen              # Запуск
sheptun listen -m small     # С моделью small
sheptun test-mic            # Проверка микрофона
sheptun list-commands       # Список команд
```

## Голосовые команды

- `клод` → вводит `claude`
- `таб`, `энтер`, `эскейп` → клавиши
- `слэш хелп` → `/help` + Enter
- `скажи <текст>` → вводит текст
- `стоп` → останавливает

Настройка команд: `config/commands.yaml`
