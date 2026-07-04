# Память проекта

Долговременная память проекта. Скилл `/kb-remember` дописывает сюда воспоминания.
Перед работой читается этот индекс; конкретная тема подгружается по необходимости.

## Темы

- [Деплой и перезапуск](deploy.md) — правки Python-кода: хватает `sheptun restart`, пересборка `.app` не нужна
- [Вставка текста](text-insertion.md) — метод по умолчанию clipboard/Cmd+V, ConcealedType против CopyClip, снапшот всех типов буфера, trailing-пробел
- [LLM Enhancement](llm-enhancement.md) — локальное причёсывание транскрипта (VoiceInk-стиль): reasoning off, галлюцинация терминов, LM Studio vs macOS 13.7, наш Whisper на MLX/GPU
- [Анализатор логов](log-analyzer.md) — грабли прокси/стрима (непоследовательные индексы блоков, out=0/max_tokens), fallback-разбор ответа без [ ], флаг LOG_PROMPTS, не коммитить replacements.yaml от тестов
