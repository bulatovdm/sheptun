# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Sheptun is a voice-controlled terminal application for Russian language. Uses local Whisper speech recognition to control terminal via voice on macOS.

## Commands

```bash
# Development
pip install -e ".[dev]"
ruff check src tests && ruff format src tests && mypy src
pytest

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
├── cli.py          # Typer CLI
├── menubar.py      # macOS menubar app (rumps)
├── engine.py       # VoiceEngine orchestrator
├── audio.py        # Audio recording + VAD
├── recognition.py  # WhisperRecognizer
├── commands.py     # CommandParser + YAML loader
├── keyboard.py     # MacOSKeyboardSender (Quartz)
├── settings.py     # Settings from .env
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
