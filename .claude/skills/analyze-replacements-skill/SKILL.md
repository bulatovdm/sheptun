---
name: analyze-replacements-skill
description: Skill-based twin of the `sheptun analyze-replacements` CLI — analyses logs/sheptun.log and appends word-replacement rules to replacements.yaml, but uses Claude Code subagents (default haiku) instead of the Anthropic SDK. Runs in iterations with a persistent checkpoint (dataset/skill_analyzer_state.json), several batches in parallel, reusing the SAME prompts and the same window/validation pipeline as the CLI. Use when the user asks to analyse the log for replacements via the skill / subagents, or to continue a previous skill run.
---

# Analyze replacements (skill-версия)

То же, что CLI `sheptun analyze-replacements`, но модель — не HTTP-запрос через SDK,
а субагенты Claude Code. Детерминистская часть (окна контекста, промпты, валидация,
запись в YAML, чекпоинт) переиспользуется из кода проекта — расхождений с CLI нет.

Оркестратор (текущая модель сессии) только раздаёт батчи и собирает ответы.
Анализ каждого батча делает субагент на дешёвой модели.

## Конфигурация

Значения по умолчанию — в `config.yaml` рядом с этим файлом. **Прочитай его в начале прогона**
и используй как дефолты. Аргументы вызова скилла перекрывают конфиг.

Ключи: `subagent_model`, `subagent_effort`, `max_batches`, `concurrency`, `batch_size`,
`context_lines`, `min_freq`, `min_confidence`.

Субагент — тип `replacement-batch` (`.claude/agents/replacement-batch/AGENT.md`).
Там же его `effort` и `maxTurns`: Agent tool параметр `effort` не принимает, поэтому
менять reasoning effort нужно во frontmatter агента, а не в `config.yaml`.
`model` из `config.yaml` передавай в вызов Agent — он перекрывает модель агента.

Аргументы из запроса пользователя (в свободной форме, распознавай по смыслу):
`5 итераций` → `max_batches=5`, `моделью sonnet` → `subagent_model=sonnet`,
`с начала` / `заново` → `--full`, `сбрось чекпоинт` → шаг «Сброс».

## Состояние

- Чекпоинт: `dataset/skill_analyzer_state.json` (**свой**, не трогает чекпоинт CLI
  `dataset/analyzer_state.json`).
- Отчёт прогона: `tmp/replacements.skill.<timestamp>.yaml` (новый файл на прогон).
- Боевой файл: `src/sheptun/config/replacements.yaml` — правила дописываются **всегда**,
  сразу после каждого батча. Пользователь смотрит их глазами и коммитит сам.
  Ничего в этом файле не удаляй и не переписывай.

## Процедура

### 0. Подготовка

```bash
python -m sheptun.skill_analyzer status
```

Покажи пользователю текущую позицию и число прогонов.

### 1. План батчей

```bash
python -m sheptun.skill_analyzer plan \
  --max-batches <max_batches> \
  --batch-size <batch_size> \
  --context <context_lines> \
  --min-freq <min_freq> \
  > tmp/skill_plan.json
```

**Порядок обхода — от свежих к старым** (по умолчанию): свежая речь и есть источник новых
ошибок ASR, ранняя история уже разобрана прошлыми прогонами. Позиция в чекпоинте — это
число окон, пройденных **с конца лога**, так что продолжение прогона уходит всё глубже
в прошлое. Хронологический порядок (как у CLI) — флаг `--oldest-first`.

Добавь `--full`, если просили начать сначала (то есть снова с самых свежих);
`--start N` — начать с конкретной позиции.

Ответ — компактный JSON: `full_total`, `start_position` и `batches[]`, где у каждого батча
`index`, `start`, `end` и `task` — путь к файлу `tmp/skill_batch_<index>.txt` с полностью
готовым заданием (критерии из `prompts/replacements_system.md` + фрагменты лога — ровно те
же промпты, что у CLI).

**Файлы заданий НЕ читай.** Они большие (десятки КБ каждый) и нужны только субагенту.
Оркестратор оперирует лишь путями и позициями — в этом весь смысл схемы: тяжёлый текст
не проходит через дорогую модель.

Если `batches` пуст — новых строк нет, сообщи об этом и заверши.

Скажи пользователю: сколько батчей, с какой позиции, из скольких окон всего.

### 2. Прогон батчей (итерации)

Обрабатывай батчи **по порядку**, группами по `concurrency` штук.
Для каждой группы запусти субагентов **одним сообщением** (чтобы шли параллельно),
`subagent_type: "replacement-batch"`, `model:` из конфига (`subagent_model`),
`run_in_background: false`, и промпт вида:

```
Прочитай файл <абсолютный путь из поля task> и выполни задание из него.

Затем запиши свой ответ (только JSON-массив) в файл tmp/skill_reply_<index>.json
инструментом Write, и верни этот же JSON финальным текстом.

Формат: [{"old": "...", "new": "...", "confidence": "high|medium|low", "reason": "..."}]
без markdown-обёртки и без пояснений. Если замен нет — верни [].
Кроме чтения файла задания и записи ответа, никаких инструментов не используй.
```

Файл ответа пишет сам субагент — оркестратору сохранять ничего не нужно. Если файл
`tmp/skill_reply_<index>.json` не появился, запиши в него финальный текст субагента
**как есть**, без правок и без «починки» JSON: парсер на стороне `commit` устойчив
к мусору вокруг массива.

### 3. Коммит батча

**Строго по возрастанию `index`** (чекпоинт двигается только по непрерывному префиксу —
не коммить батч N+1, пока не закоммичен N):

```bash
python -m sheptun.skill_analyzer commit \
  --reply tmp/skill_reply_<index>.json \
  --report tmp/replacements.skill.<timestamp>.yaml \
  --position <end батча> \
  --min-confidence <min_confidence>
```

Команда сама: нормализует ответ, отбрасывает выдуманные `old` (которых нет ни в одной
строке лога), берёт реальную частоту из лога, режет по порогу уверенности, дедуплицирует
против уже существующих правил, дописывает в отчёт и в `replacements.yaml`, двигает чекпоинт.

Из её JSON-ответа покажи пользователю строкой прогресса: `батч i/N`, принятые правила
(`old → new`, conf, freq) и сколько отброшено.

Если субагент упал или вернул мусор — не коммить этот батч, останови прогон на нём
(чекпоинт останется на последнем успешном) и скажи пользователю, где встали.

### 4. Итог

Сводка: обработано батчей и окон, новая позиция `X / full_total`, сколько правил добавлено
в `replacements.yaml`, путь к отчёту. Напомни, что стоит проверить новые правила скиллом
`review-replacements` перед коммитом.

### Сброс

```bash
python -m sheptun.skill_analyzer reset
```

## Границы

- Не коммить и не пушить изменения — пользователь ревьюит сам.
- Не редактируй промпты в `src/sheptun/prompts/` в рамках прогона.
- Не изобретай свои правила как оркестратор: правила приходят только от субагентов
  и проходят валидацию `commit`.
