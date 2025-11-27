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
sheptun list-models         # Show downloaded Whisper models
sheptun cleanup-models      # Remove unused models
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
├── i18n.py         # Russian translations
├── app_builder.py  # macOS .app bundle builder
└── types.py        # Protocols, dataclasses, enums
```

**Data flow:** Microphone → VAD → Whisper → CommandParser → KeyboardSender

## Configuration

`.env` file:
```bash
SHEPTUN_MODEL=medium         # tiny, base, small, medium, large
SHEPTUN_SILENCE_DURATION=0.3 # Pause to detect end of phrase
SHEPTUN_DEBUG=false
```

Command config: `./sheptun.yaml` or `~/.config/sheptun/commands.yaml`

## Key Patterns

- Protocols in `types.py` for dependency injection
- VAD (Voice Activity Detection) with energy threshold + silence duration
- Quartz CGEventCreateKeyboardEvent for keyboard simulation
- Settings loaded once at import via dotenv (restart needed for changes)

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

## Git Commits

Do not add Co-Authored-By or emoji badges to commit messages.
