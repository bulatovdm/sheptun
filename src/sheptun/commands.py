import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sheptun.types import Action, ActionType

PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
SLASH_PREFIXES = ("слэш ", "слышь ", "слеш ")


def _build_replacement_patterns(
    replacements: dict[str, str],
) -> list[tuple[re.Pattern[str], str]]:
    patterns: list[tuple[re.Pattern[str], str]] = []
    for old, new in replacements.items():
        pattern = re.compile(r"\b" + re.escape(old) + r"\b", re.IGNORECASE | re.UNICODE)
        patterns.append((pattern, new))
    return patterns


@dataclass
class CommandConfig:
    control_commands: dict[str, Action] = field(default_factory=dict)
    stop_commands: set[str] = field(default_factory=set)
    slash_commands: dict[str, str] = field(default_factory=dict)
    dictation_prefixes: list[str] = field(default_factory=list)
    help_commands: set[str] = field(default_factory=set)
    replacements: dict[str, str] = field(default_factory=dict)


class CommandConfigLoader:
    @staticmethod
    def load(config_path: Path) -> CommandConfig:
        with config_path.open(encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        return CommandConfigLoader._parse_config(raw_config)

    @staticmethod
    def _parse_config(raw_config: dict[str, Any]) -> CommandConfig:
        control_commands = CommandConfigLoader._parse_control_commands(
            raw_config.get("control_commands", {})
        )
        stop_commands = set(raw_config.get("stop_commands", []))
        slash_commands = raw_config.get("slash_commands", {})
        dictation_prefixes = raw_config.get("dictation_prefixes", [])
        help_commands = set(raw_config.get("help_commands", []))
        replacements = raw_config.get("replacements", {})

        return CommandConfig(
            control_commands=control_commands,
            stop_commands=stop_commands,
            slash_commands=slash_commands,
            dictation_prefixes=dictation_prefixes,
            help_commands=help_commands,
            replacements=replacements,
        )

    @staticmethod
    def _parse_control_commands(raw_commands: dict[str, dict[str, Any]]) -> dict[str, Action]:
        commands: dict[str, Action] = {}

        for trigger, action_data in raw_commands.items():
            action = CommandConfigLoader._parse_action(action_data)
            if action is not None:
                commands[trigger.lower()] = action

        return commands

    @staticmethod
    def _parse_action(action_data: dict[str, Any]) -> Action | None:
        action_type_str = action_data.get("type", "")
        value = action_data.get("value", "")

        type_mapping = {
            "text": ActionType.TEXT,
            "key": ActionType.KEY,
            "hotkey": ActionType.HOTKEY,
        }

        action_type = type_mapping.get(action_type_str)
        if action_type is None:
            return None

        return Action(action_type=action_type, value=value)


class CommandParser:
    def __init__(self, config: CommandConfig) -> None:
        self._config = config
        self._replacement_patterns = _build_replacement_patterns(config.replacements)

    @classmethod
    def from_config_file(cls, config_path: Path) -> "CommandParser":
        config = CommandConfigLoader.load(config_path)
        return cls(config)

    def apply_replacements(self, text: str) -> str:
        for pattern, replacement in self._replacement_patterns:
            text = pattern.sub(replacement, text)
        return text

    def parse(self, text: str) -> Action | None:
        normalized = self._normalize_text(text)

        if not normalized:
            return None

        stop_action = self._try_parse_stop(normalized)
        if stop_action is not None:
            return stop_action

        help_action = self._try_parse_help(normalized)
        if help_action is not None:
            return help_action

        control_action = self._try_parse_control(normalized)
        if control_action is not None:
            return control_action

        slash_action = self._try_parse_slash(normalized)
        if slash_action is not None:
            return slash_action

        dictation_action = self._try_parse_dictation(normalized)
        if dictation_action is not None:
            return dictation_action

        return Action(action_type=ActionType.TEXT, value=text.strip())

    def _normalize_text(self, text: str) -> str:
        cleaned = PUNCTUATION_PATTERN.sub("", text)
        return cleaned.lower().strip()

    def _try_parse_stop(self, normalized: str) -> Action | None:
        if normalized in self._config.stop_commands:
            return Action(action_type=ActionType.STOP, value="")
        return None

    def _try_parse_help(self, normalized: str) -> Action | None:
        if normalized in self._config.help_commands:
            return Action(action_type=ActionType.HELP, value="")
        return None

    def _try_parse_control(self, normalized: str) -> Action | None:
        return self._config.control_commands.get(normalized)

    def _try_parse_slash(self, normalized: str) -> Action | None:
        command_part: str | None = None

        for prefix in SLASH_PREFIXES:
            if normalized.startswith(prefix):
                command_part = normalized[len(prefix) :].strip()
                break

        if command_part is not None:
            slash_value = self._config.slash_commands.get(command_part)
            if slash_value is not None:
                return Action(action_type=ActionType.SLASH, value=slash_value)
            return Action(action_type=ActionType.SLASH, value=f"/{command_part}")

        slash_value = self._config.slash_commands.get(normalized)
        if slash_value is not None:
            return Action(action_type=ActionType.SLASH, value=slash_value)

        return None

    def _try_parse_dictation(self, normalized: str) -> Action | None:
        for prefix in self._config.dictation_prefixes:
            if normalized.startswith(prefix + " "):
                text = normalized[len(prefix) + 1 :].strip()
                if text:
                    return Action(action_type=ActionType.TEXT, value=text)
                return None
        return None
