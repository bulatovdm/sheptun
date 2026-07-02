# Benchmarks — корректоры текста

Сравнивает корректоры ASR-вывода (SAGE, JamSpell, …) на **реальных логах Sheptun**,
с упором на главную опасность для смешанной ru/en речи: **урон по англотерминам**
(корректор «чинит» валидный `docker`→`уокер`, `git`→`гит` — хуже, чем оставить опечатку).

## Установка

```bash
pip install -e ".[bench]"          # SAGE (transformers) + torch
```

JamSpell ставится отдельно — его wheel собирается только со **SWIG 3**, а в системе
обычно SWIG 4. Обход (ослабить проверку версии в setup.py):

```bash
pip download jamspell --no-binary :all: --no-deps -d /tmp/js && cd /tmp/js
tar xzf jamspell-*.tar.gz && cd jamspell-*/
# заменить строку assert ...'SWIG Version 3'... на pass
sed -i '' "s/assert subprocess.check_output(\[swigBinary, \"-version\"\]).*/pass/" setup.py
pip install .
# русская модель (39 МБ):
curl -sL -o ru.tar.gz https://raw.githubusercontent.com/bakwc/JamSpell-models/master/ru.tar.gz
tar xzf ru.tar.gz  # → ru_small.bin
export SHEPTUN_BENCH_JAMSPELL_MODEL=$PWD/ru_small.bin
```

## Запуск

```bash
# SAGE на 200 реальных строк из логов (дефолт)
python -m benchmarks run --correctors noop,sage --source log --count 200

# добавить JamSpell (нужен SHEPTUN_BENCH_JAMSPELL_MODEL)
python -m benchmarks run --correctors noop,jamspell,sage

# точность против эталона из replacements.yaml (искажение → правильно)
python -m benchmarks run --correctors sage --source replacements --count 300
```

## Метрики

| Колонка | Смысл |
|---|---|
| Изменено | доля строк, где вывод ≠ вход |
| **Повреждено** | доля строк, где пропал англотермин (латиница или транслит) — **главный риск** |
| Потеряно lat/термы | сколько латинских слов / ru-транслитов исчезло суммарно |
| Точность | exact-match против эталона (только `--source replacements`) |
| мс/строку | скорость |

## Архитектура (расширяемо)

- `types.py` — `Corrector` (Protocol), `Sample`, `CorrectionResult`, `*Report` (frozen dataclasses).
- `correctors/` — реализации + реестр по имени (`create`/`available`). **Добавить корректор:**
  новый файл с классом (свойство `name`, `setup()`, `correct()`) → регистрация в `correctors/__init__.py`.
- `samples.py` — выборка из логов (`from_log`, без эталона) или из replacements.yaml (`from_replacements`, с эталоном).
- `metrics.py` — урон по англотерминам, exact-match. Список защищаемых транслитов — `DEFAULT_TERMS`.
- `runner.py` — оркестратор. `cli.py` — Typer-CLI.

Тесты: `tests/benchmarks/` (метрики, выборка, runner, реестр — без тяжёлых моделей, через test-double корректоры).

## Результаты

Сохраняются в [`docs/benchmark-correctors.md`](../docs/benchmark-correctors.md) — прогоняй периодически и обновляй.
