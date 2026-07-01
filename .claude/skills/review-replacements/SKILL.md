---
name: review-replacements
description: Reviews word-replacement rules in sheptun's replacements.yaml for correctness after a new analysis pass. Flags real-word keys, punctuation, duplicates, dubious translations and language-mix; produces a check file plus a review report; then asks the user whether to apply the fixes and, on yes, removes the bad rules from replacements.yaml. Also proposes concrete additions to the generation prompt. Use after running `sheptun analyze-replacements` or when the user asks to review/audit replacement rules or improve the analyzer prompt.
---

# Review replacements

Критически проверяет правила автозамены в `src/sheptun/config/replacements.yaml`, находит
ошибочные, оформляет ревью и — с подтверждения пользователя — вычищает их из боевого файла.
Плюс предлагает, как усилить промпт генерации, чтобы такие ошибки не появлялись впредь.

Правила применяются как ГЛОБАЛЬНАЯ регистронезависимая замена целых слов (`\bold\b` -> `new`)
ко всей будущей распознанной речи. Одно ложное правило ломает диктовку навсегда — поэтому
критерий строгий: при любом сомнении правило считается ошибочным.

## Когда запускать

- После `sheptun analyze-replacements` (когда добавились новые правила).
- Когда пользователь просит проверить/провести аудит `replacements.yaml`.
- Когда просят предложить улучшение промпта анализатора.

## Процесс

### Шаг 1. Определить, что ревьюить
Новые правила — те, что снабжены комментарием `# freq=..., conf=...` (их добавляет анализатор).
Старые (без комментария) обычно уже выверены — по умолчанию не трогай, если пользователь не
попросил проверить всё.

```bash
grep -nE '# freq=' src/sheptun/config/replacements.yaml | wc -l   # сколько новых правил
```

### Шаг 2. Проверить каждое правило по чек-листу
Полные критерии и классы ошибок — в [CRITERIA.md](CRITERIA.md). Прочти его перед проверкой.
Кратко: правило ОШИБОЧНО, если `old` — реальное слово, содержит пунктуацию, дублирует другое,
даёт сомнительный перевод, смешивает язык цели, ИЛИ его комментарий сам говорит «пропускаю».

Для больших объёмов (сотни новых правил) распараллель проверку по диапазонам строк через
несколько субагентов, каждому дай CRITERIA.md как единый чек-лист, затем собери вердикты.

### Шаг 3. Проверить на чувствительные данные
Прогони скан из [SENSITIVE.md](SENSITIVE.md). Секреты/личные данные не должны попадать в правила.

### Шаг 4. Сгенерировать артефакты
1. `replacements.check.yaml` — копия конфига, где ОШИБОЧНЫЕ правила **закомментированы** строкой
   `# ` и над каждым добавлена причина `# >>> WRONG: <причина>`. Корректные — как есть.
   Файл должен остаться валидным YAML. Скрипт: [make_check.py](scripts/make_check.py).
2. `REPLACEMENTS_REVIEW.md` — отчёт: сводка по классам ошибок + таблица «строка / правило / почему».
3. Секция с предложениями по промпту (см. Шаг 6).

Все три файла — временные артефакты ревью, они в `.gitignore` (`replacements.check.yaml`,
`REPLACEMENTS_REVIEW.md`). Боевой `replacements.yaml` на этом шаге НЕ трогай.

### Шаг 5. Спросить пользователя и применить
Покажи сводку (сколько ошибочных, по классам). **Спроси у пользователя, применять ли** —
удалить ли помеченные WRONG-правила из `src/sheptun/config/replacements.yaml`.
- Если пользователь согласился на все — удали все помеченные строки.
- Если пользователь отобрал часть (проходится по `replacements.check.yaml` сам) — удали только
  подтверждённые им.
- Если отказался — ничего не меняй в боевом файле, оставь артефакты для ручного разбора.

После удаления проверь, что файл остаётся валидным YAML:
```bash
python3 -c "import yaml,pathlib; print('rules:', len(yaml.safe_load(pathlib.Path('src/sheptun/config/replacements.yaml').read_text(encoding='utf-8'))))"
```

### Шаг 6. Предложить корректировку промпта
На основе найденных классов ошибок предложи конкретные добавления в
`src/sheptun/prompts/replacements_system.md` (и при необходимости в `replacements_verify.md`).
Шаблон и привязка «класс ошибки → правило промпта» — в [PROMPT_TUNING.md](PROMPT_TUNING.md).
НЕ меняй промпт молча — покажи предложения и спроси подтверждения.

## Важно
- Артефакты ревью НЕ коммить (они в `.gitignore`).
- Боевой `replacements.yaml` меняй ТОЛЬКО после явного согласия пользователя.
- Не удаляй старые (без `# freq=`) правила без отдельной просьбы.
