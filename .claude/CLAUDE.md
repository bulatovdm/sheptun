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
├── dataset.py      # DatasetRecorder for fine-tuning data collection
├── i18n.py         # Russian translations
├── verification.py # Transcript verification via Claude Agent SDK
├── log_analyzer.py # LLM log analysis → replacement suggestions (Anthropic SDK)
├── prompts/        # Prompt templates as .md files + load_prompt() loader
├── app_builder.py  # macOS .app bundle builder
└── types.py        # Protocols, dataclasses, enums (AppState)
```

**Data flow:** Microphone → VAD → Whisper → Hallucination filter → Spell correction → Word replacements → CommandParser → KeyboardSender

## Configuration

`.env` file:
```bash
SHEPTUN_MODEL=medium         # tiny, base, small, medium, large
SHEPTUN_SILENCE_DURATION=0.3 # Pause to detect end of phrase
SHEPTUN_DEBUG=false
```

Command config: `./sheptun.yaml` or `~/.config/sheptun/commands.yaml`

Log analyzer (`sheptun analyze-replacements`, extra `[llm]`): `SHEPTUN_ANTHROPIC_BASE_URL`, `SHEPTUN_ANTHROPIC_API_KEY`, and `SHEPTUN_ANALYZER_*` (model, context, batch, iterations, min_freq, min_confidence, user_agent).

## Debugging

- App logs: `./logs/sheptun.log` (relative to project root)
- Crash reports: `~/Library/Logs/DiagnosticReports/Python-*.ips` (look for `com.sheptun.menubar`)

## Log Analyzer (`log_analyzer.py`)

LLM pipeline behind `sheptun analyze-replacements` — decomposed by SRP, each stage independently configurable:

`LogParser` (extract Recognized lines only, drop noise) → `ContextWindowBuilder` (±N neighbouring Recognized lines per target, dedup + frequency, since/until filter) → `WindowBatcher` → `AnthropicClient` (Anthropic SDK, custom base_url/api_key from env) → `ReplacementAnalyzer` (orchestrates, incremental dedup) → `SuggestionWriter` (report and/or apply).

- Uses the **Python `anthropic` SDK** with a custom User-Agent (default SDK UA gets blocked by some proxies; override via `SHEPTUN_ANALYZER_USER_AGENT`). No structured-output/`output_config.format` — the JSON shape is required in the prompt and parsed robustly (`_extract_items`).
- **Incremental by default:** checkpoint (last processed timestamp) in `dataset/analyzer_state.json`. `--since`/`--until` for explicit ranges, `--full` to ignore checkpoint, `--max-iterations` to cap model requests per run (processes windows chronologically so the checkpoint advances without gaps).
- Suggestions are written/applied **after each batch** (crash-safe), with live per-batch progress.
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

- **review-replacements** (`.claude/skills/review-replacements/`) — reviews new rules in `replacements.yaml` after an analysis pass: flags real-word keys / punctuation / duplicates / dubious translations / language-mix, writes `replacements.check.yaml` + `REPLACEMENTS_REVIEW.md`, asks the user before removing bad rules from the live file, and proposes prompt improvements. Bundles CRITERIA.md, SENSITIVE.md, PROMPT_TUNING.md, scripts/make_check.py. Trigger by asking to review/audit replacement rules or after `sheptun analyze-replacements`.

## Git Commits

Do not add Co-Authored-By or emoji badges to commit messages.
