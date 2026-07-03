# Деплой и перезапуск

## Изменения Python-кода: достаточно restart, пересборка не нужна
`.app`-бандл — тонкая обёртка: лаунчер `MacOS/sheptun` делает `cd` в проект, активирует venv и запускает `python -m sheptun.menubar`, то есть код читается **напрямую из `src/sheptun/`** (пакет установлен editable, `pip install -e`). Поэтому после правок Python-кода достаточно `sheptun restart` — **пересобирать `sheptun install-app` не нужно**. Пересборка требуется только при смене иконки, `Info.plist`, имени бандла или самого лаунчер-скрипта.
Связано: `src/sheptun/app_builder.py:51` (write_executable), команда `sheptun restart`.

## Пробел между фрагментами — trailing-модель, НЕ подавлять после точки/?/!
Вставка пробела сделана как **trailing space** (пробел ПОСЛЕ каждого фрагмента, как у VoiceInk), а не leading. Живёт в `_prepare_text` (`src/sheptun/engine.py`), константа `_NO_TRAILING_SPACE_AFTER = frozenset("\n ")`. Грабли: НЕ добавлять `.!?…` в исключения — Whisper сам ставит точку/`?` в конце фразы, и без trailing-пробела следующий фрагмент липнет вплотную («Привет.Что думаешь?»). Пробел подавляем только если фрагмент уже кончается пробелом или `\n`. Старый leading-подход читал позицию курсора через AX (`AXSelectedTextRange`) и давал баг «пробел в начале новой строки» в Apple Notes — удалён вместе с `get_cursor_position`/`has_text_before_cursor`.
Связано: `src/sheptun/engine.py` (`_prepare_text`, `_NO_TRAILING_SPACE_AFTER`), `tests/test_engine.py` (test_prepare_text_*).
