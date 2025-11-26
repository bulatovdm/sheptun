# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sheptun is a voice-controlled terminal application for Russian language commands. It uses local Whisper speech recognition to control the terminal via voice, primarily designed for macOS.

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the application
sheptun listen                    # Start listening with default settings
sheptun listen -m small           # Use smaller/larger Whisper model
sheptun listen -c path/to/config  # Custom config file
sheptun test-mic                  # Test microphone
sheptun list-commands             # Show available commands

# Code quality
ruff check src tests              # Lint
ruff format src tests             # Format
mypy src                          # Type check (strict mode)

# Testing
pytest                            # Run all tests
pytest tests/test_commands.py     # Run specific test file
pytest -k "test_parse"            # Run tests matching pattern
```

## Architecture

```
src/sheptun/
├── cli.py          # Typer CLI entry point
├── engine.py       # VoiceEngine orchestrator
├── recognition.py  # WhisperRecognizer (speech-to-text)
├── commands.py     # CommandParser + YAML config loader
├── keyboard.py     # MacOSKeyboardSender (Quartz-based)
├── audio.py        # Audio recording + VAD
├── status.py       # Status indicators (Rich-based)
└── types.py        # Protocols, dataclasses, enums
```

**Data flow:** Microphone → VAD (silence detection) → Whisper → CommandParser → KeyboardSender

**Command config lookup order:**
1. `./sheptun.yaml` (project-local)
2. `~/.config/sheptun/commands.yaml` (user global)
3. Built-in default (`src/sheptun/config/commands.yaml`)

## Key Design Decisions

- **Protocols for interfaces:** `types.py` defines Protocol classes for dependency injection
- **Action types:** TEXT (type string), KEY (single key), HOTKEY (modifier combo), SLASH (Claude Code commands), STOP (exit)
- **VAD-based recording:** ContinuousAudioRecorder uses energy threshold + silence duration to detect speech boundaries
- **macOS keyboard:** Uses Quartz CGEventCreateKeyboardEvent for reliable keystroke simulation

## Adding New Commands

Create `~/.config/sheptun/commands.yaml` or `./sheptun.yaml`:
```yaml
control_commands:
  "новая команда": { type: "key", value: "f5" }
```

Types: `text` (type string), `key` (press key), `hotkey` (modifier+key combo)
