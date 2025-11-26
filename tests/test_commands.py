from pathlib import Path

import pytest

from sheptun.commands import CommandConfig, CommandParser
from sheptun.types import Action, ActionType


@pytest.fixture
def command_config() -> CommandConfig:
    return CommandConfig(
        control_commands={
            "клод": Action(ActionType.TEXT, "claude"),
            "таб": Action(ActionType.KEY, "tab"),
            "шифт таб": Action(ActionType.HOTKEY, ["shift", "tab"]),
            "энтер": Action(ActionType.KEY, "return"),
        },
        stop_commands={"стоп", "хватит"},
        slash_commands={
            "хелп": "/help",
            "клир": "/clear",
        },
        dictation_prefixes=["скажи", "введи"],
    )


@pytest.fixture
def parser(command_config: CommandConfig) -> CommandParser:
    return CommandParser(command_config)


class TestCommandParser:
    def test_parse_control_command(self, parser: CommandParser) -> None:
        action = parser.parse("клод")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "claude"

    def test_parse_control_command_case_insensitive(self, parser: CommandParser) -> None:
        action = parser.parse("КЛОД")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "claude"

    def test_parse_key_command(self, parser: CommandParser) -> None:
        action = parser.parse("таб")

        assert action is not None
        assert action.action_type == ActionType.KEY
        assert action.value == "tab"

    def test_parse_hotkey_command(self, parser: CommandParser) -> None:
        action = parser.parse("шифт таб")

        assert action is not None
        assert action.action_type == ActionType.HOTKEY
        assert action.value == ["shift", "tab"]

    def test_parse_stop_command(self, parser: CommandParser) -> None:
        action = parser.parse("стоп")

        assert action is not None
        assert action.action_type == ActionType.STOP

    def test_parse_stop_command_variant(self, parser: CommandParser) -> None:
        action = parser.parse("хватит")

        assert action is not None
        assert action.action_type == ActionType.STOP

    def test_parse_slash_command_known(self, parser: CommandParser) -> None:
        action = parser.parse("слэш хелп")

        assert action is not None
        assert action.action_type == ActionType.SLASH
        assert action.value == "/help"

    def test_parse_slash_command_unknown(self, parser: CommandParser) -> None:
        action = parser.parse("слэш фу")

        assert action is not None
        assert action.action_type == ActionType.SLASH
        assert action.value == "/фу"

    def test_parse_dictation_with_prefix(self, parser: CommandParser) -> None:
        action = parser.parse("скажи привет мир")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "привет мир"

    def test_parse_dictation_with_another_prefix(self, parser: CommandParser) -> None:
        action = parser.parse("введи тестовый текст")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "тестовый текст"

    def test_parse_unrecognized_as_text(self, parser: CommandParser) -> None:
        action = parser.parse("произвольный текст")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "произвольный текст"

    def test_parse_empty_text(self, parser: CommandParser) -> None:
        action = parser.parse("")

        assert action is None

    def test_parse_whitespace_only(self, parser: CommandParser) -> None:
        action = parser.parse("   ")

        assert action is None

    def test_parse_preserves_original_case_for_text(self, parser: CommandParser) -> None:
        action = parser.parse("Hello World")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "Hello World"
