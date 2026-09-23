# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Sheptun is a voice-controlled terminal application for Russian language. Uses local Whisper speech recognition to control terminal via voice on macOS.

## Commands

```bash
# Development
pip install -e ".[dev]"
pytest

# Linting & Type Checking (run all before commit)
ruff check src tests        # Linting (style, imports, errors)
ruff format src tests       # Auto-format code
mypy src                    # Strict type checking for src
pyright src tests           # Type checking like VS Code/Pylance

# All checks in one command
ruff check src tests && mypy src && pyright src tests

# Application
sheptun listen              # CLI mode
sheptun install-app         # Create menubar app
sheptun restart             # Restart menubar app
sheptun list-models         # Show all cached models with sizes
sheptun cleanup-models      # Remove unused models (Whisper + HuggingFace cache)
sheptun clear-dataset       # Clear dataset for fine-tuning

# Suggest word replacements from logs via Anthropic SDK (needs pip install -e ".[llm]")
sheptun analyze-replacements --min-confidence high          # incremental, report only
sheptun analyze-replacements --max-iterations 5 --apply     # process N batches, write to replacements.yaml
sheptun analyze-replacements --since 2026-06-01 --dry-run   # count windows for a date range
sheptun analyze-replacements --reset-state                  # clear the incremental checkpoint

# KenLM n-gram LM for the GigaAM decoder
./scripts/build_kenlm.sh    # once: lmplz/build_binary → tools/kenlm/bin (brew cmake boost)
sheptun build-lm            # → models/gigaam-lm.bin
sheptun train-tagger        # term-correcting network (MLX, ~13 min) → models/term-tagger

```

## Architecture

```
src/sheptun/
├── cli.py          # Typer CLI entry point
├── menubar.py      # macOS menubar app (rumps)
├── engine.py       # VoiceEngine orchestrator (BaseVoiceEngine)
├── audio.py        # Audio recording + VAD (EnergyVAD, SileroVAD)
├── recognition.py  # WhisperRecognizer
├── commands.py     # CommandParser + YAML config loader
├── keyboard.py     # MacOSKeyboardSender (Quartz), FocusAwareKeyboardSender
├── focus.py        # FocusTracker for PTT mode (NSWorkspace)
├── hotkeys.py      # HotkeyManager (pynput) for global hotkeys
├── status.py       # Console status indicators (Rich)
├── settings.py     # Settings from .env
├── formatting.py   # TechnicalFormatter: spoken symbols → code (snake_case, casing)
├── text_cleanup.py # TextCleaner: collapse seam duplicates (..env→.env, repeated words)
├── dataset.py      # DatasetRecorder for fine-tuning data collection
├── i18n.py         # Russian translations
├── verification.py # Transcript verification via Claude Agent SDK
├── log_analyzer.py # LLM log analysis → replacement suggestions (Anthropic SDK)
├── gigaam.py       # GigaAM recognizer; CTC beam search with hotwords + KenLM
├── lm_corpus.py    # Corpus for the GigaAM LM: log + current replacements + verification.db
├── ngram_lm.py     # KenLM training (lmplz) + normalized LM scorer for pyctcdecode
├── term_tagger.py  # Term tagger: per-word KEEP/DELETE/term labels → corrected text (no MLX)
├── tagger_data.py  # Tagger examples: rule matches in the log + synthetic ASR distortions
├── tagger_model.py # Tagger network (char-CNN + Transformer, MLX) and its training loop
├── prompts/        # Prompt templates as .md files + load_prompt() loader
├── app_builder.py  # macOS .app bundle builder
└── types.py        # Protocols, dataclasses, enums (AppState)
```

**Data flow:** Microphone → VAD → ASR (GigaAM + KenLM + hotwords) → Hallucination filter → Term tagger (`term_tagger.py`, optional) → Word replacements (`replacements.yaml`) → Technical formatting (`formatting.py`) → Text cleanup (`text_cleanup.py`, fillers + duplicates) → CommandParser → KeyboardSender

## Configuration

`.env` file:
```bash
SHEPTUN_MODEL=medium         # tiny, base, small, medium, large
SHEPTUN_SILENCE_DURATION=0.3 # Pause to detect end of phrase
SHEPTUN_DEBUG=false
```

GigaAM beam search (mlx + CTC only): `SHEPTUN_GIGAAM_HOTWORDS` / `_HOTWORDS_LIMIT` / `_HOTWORD_WEIGHT`, KenLM via `SHEPTUN_GIGAAM_LM_PATH` (+ `_LM_ALPHA` 0.3, `_LM_BETA` 1.0, `_LM_UNK_OFFSET` -5). Best on the 100-phrase testset: LM + 652 hotwords, weight 30 → CER 7%/9%, EPI 49%, ~95ms (greedy 9%/11%, EPI 31%) (see `docs/asr-benchmark-2026-09.md`).

Term tagger: `SHEPTUN_TAGGER_PATH=models/term-tagger` (+ `SHEPTUN_TAGGER_THRESHOLD` 0.5) — a 2.3M-param MLX network that picks a term from the replacement values for each ASR-mangled word (~1ms/phrase). Retrain after new replacement rules.

Command config: `./sheptun.yaml` or `~/.config/sheptun/commands.yaml`

Log analyzer (`sheptun analyze-replacements`, extra `[llm]`): `SHEPTUN_ANTHROPIC_BASE_URL`, `SHEPTUN_ANTHROPIC_API_KEY`, and `SHEPTUN_ANALYZER_*` (model, context, batch, concurrency, max_tokens, thinking, effort, iterations, min_freq, min_confidence, stream, user_agent).

## Debugging

- App logs: `./logs/sheptun.log` (relative to project root)
- Crash reports: `~/Library/Logs/DiagnosticReports/Python-*.ips` (look for `com.sheptun.menubar`)

## Log Analyzer (`log_analyzer.py`)

LLM pipeline behind `sheptun analyze-replacements` — decomposed by SRP, each stage independently configurable:

`LogParser` (extract Recognized lines only, drop noise) → `ContextWindowBuilder` (±N neighbouring Recognized lines per target, dedup + frequency, since/until filter) → `WindowBatcher` → `AnthropicClient` (Anthropic SDK, custom base_url/api_key from env) → `ReplacementAnalyzer` (orchestrates, incremental dedup) → `SuggestionWriter` (report and/or apply).

- Uses the **Python `anthropic` SDK** with a custom User-Agent (default SDK UA gets blocked by some proxies; override via `SHEPTUN_ANALYZER_USER_AGENT`). No structured-output/`output_config.format` — the JSON shape is required in the prompt and parsed robustly (`_extract_items`).
- **Incremental by default:** checkpoint (last processed timestamp) in `dataset/analyzer_state.json`. `--since`/`--until` for explicit ranges, `--full` to ignore checkpoint, `--max-iterations` to cap model requests per run (processes windows chronologically so the checkpoint advances without gaps).
- Suggestions are written/applied **after each batch** (crash-safe), with live per-batch progress.
- **Parallel batches** (`SHEPTUN_ANALYZER_CONCURRENCY`, default 5; `_analyze_parallel`): the saved position advances only over the *contiguous completed prefix* of batches, so out-of-order completion / a crash / Ctrl+C never skips a batch. Ctrl+C is swallowed, progress saved, then the CLI force-exits (`os._exit`) so uncancellable in-flight proxy requests on non-daemon threads don't hang shutdown. `concurrency=1` keeps the sequential path.
- Prompts live in `prompts/*.md`, loaded via `load_prompt()`; env knobs use the `SHEPTUN_ANALYZER_` prefix.
- The report defaults to `tmp/replacements.suggested.<timestamp>.yaml` (new file per run, never overwritten); rules carry `# freq=…, conf=…, — reason` comments. `--apply` appends the same commented rules to `replacements.yaml` (preserving existing content). `tmp/` and `dataset/` are gitignored.

## Key Patterns

- Protocols in `types.py` for dependency injection
- VAD (Voice Activity Detection) with energy threshold + silence duration
- Quartz CGEventCreateKeyboardEvent for keyboard simulation
- Hallucination filtering in `recognition.py` (configurable via `SHEPTUN_HALLUCINATIONS`)
- Settings loaded once at import via dotenv (restart needed for changes)
- Verification DB (`dataset/verification.db`) used by fine-tuning pipeline
- Prompt templates in `prompts/*.md`, loaded via `load_prompt()` (editable without touching code)

## Code Style

### Principles
- **SOLID** — single responsibility, open/closed, dependency inversion
- **DRY** — don't repeat yourself, extract common logic
- **KISS** — keep it simple, avoid over-engineering
- **YAGNI** — don't add features until needed

### Structure
- Early return — exit early to avoid deep nesting
- Small functions — split large functions into focused methods
- Flat is better than nested — max 2-3 levels of indentation
- One thing per function — each function does one thing well

### Naming
- Self-documenting code — names should explain intent
- No comments for obvious code — let the code speak
- Docstrings only when needed — don't duplicate method names
- Docstrings required for CLI commands (used by `--help`)

### Avoid
- Magic numbers — use named constants
- Unused code — delete it, don't comment out
- Deep nesting — refactor with early returns or extract methods
- Long parameter lists — use dataclasses or config objects

## Custom Agents

- **analyze-logs** — Analyzes `logs/sheptun.log` to extract vocabulary for ASR optimization. Generates recommended `SHEPTUN_INITIAL_PROMPT` and word `replacements` for `sheptun.yaml`. Invoke via `@analyze-logs`. (Distinct from the `sheptun analyze-replacements` CLI command / `log_analyzer.py`, which is a standalone Anthropic-SDK pipeline — see Log Analyzer above.)

## Skills

- **analyze-replacements-skill** (`.claude/skills/analyze-replacements-skill/`) — скилл-двойник CLI
  `sheptun analyze-replacements`: тот же лог, те же промпты (`prompts/replacements_*.md`) и та же
  валидация, но модель — субагенты Claude Code (`replacement-batch`, haiku, effort medium), а не
  Anthropic SDK. Итерации + свой чекпоинт `dataset/skill_analyzer_state.json` (не пересекается с
  чекпоинтом CLI), несколько батчей параллельно, дозапись правил в `replacements.yaml` после
  каждого батча. Обход лога — **от свежих строк к старым** (флаг `--oldest-first` вернёт
  хронологический порядок CLI); позиция чекпоинта считается от конца лога. Детерминистская
  часть — `src/sheptun/skill_analyzer.py` (`plan` / `commit` / `status` / `reset`),
  переиспользует классы `log_analyzer.py`. `plan` раскладывает готовые задания по
  `tmp/skill_batch_<N>.txt` и возвращает только пути — текст лога не проходит через контекст
  оркестратора.
  Настройки прогона — `config.yaml` рядом со SKILL.md; `effort` субагента задаётся только во
  frontmatter `.claude/agents/replacement-batch/AGENT.md` (Agent tool такого параметра не имеет).
- **review-replacements** (`.claude/skills/review-replacements/`) — reviews new rules in `replacements.yaml` after an analysis pass: flags real-word keys / punctuation / duplicates / dubious translations / language-mix, writes `replacements.check.yaml` + `REPLACEMENTS_REVIEW.md`, asks the user before removing bad rules from the live file, and proposes prompt improvements. Bundles CRITERIA.md, SENSITIVE.md, PROMPT_TUNING.md, scripts/make_check.py. Trigger by asking to review/audit replacement rules or after `sheptun analyze-replacements`.

## Git Commits

Do not add Co-Authored-By or emoji badges to commit messages.

<!-- kb-memory:start -->
## Память проекта (.kb/memory)

ОБЯЗАТЕЛЬНО в начале сессии: прочитай индекс `.kb/memory/memory.md` — оглавление долговременной
памяти проекта (темы + краткие хуки). Сами тематические файлы НЕ читай сразу — подгружай конкретную
тему из индекса только когда она нужна по текущей задаче.

Зафиксировать новое воспоминание — скилл `/kb-remember`.
<!-- kb-memory:end -->
