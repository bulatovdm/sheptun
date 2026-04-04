import tempfile
from pathlib import Path

import pytest

from sheptun.commands import CommandConfig, CommandConfigLoader, CommandParser
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

    def test_parse_stop_command_in_phrase_returns_text(self, parser: CommandParser) -> None:
        action = parser.parse("скажите стоп для выхода")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "скажите стоп для выхода"

    def test_parse_stop_command_with_punctuation(self, parser: CommandParser) -> None:
        action = parser.parse("Стоп!")

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

    def test_parse_slash_command_slish_variant(self, parser: CommandParser) -> None:
        action = parser.parse("слышь клир")

        assert action is not None
        assert action.action_type == ActionType.SLASH
        assert action.value == "/clear"

    def test_parse_slash_command_without_prefix(self, parser: CommandParser) -> None:
        action = parser.parse("клир")

        assert action is not None
        assert action.action_type == ActionType.SLASH
        assert action.value == "/clear"

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

    def test_parse_help_command(self) -> None:
        config = CommandConfig(help_commands={"помощь", "команды"})
        parser = CommandParser(config)

        action = parser.parse("помощь")

        assert action is not None
        assert action.action_type == ActionType.HELP

    def test_parse_dictation_empty_after_prefix(self) -> None:
        config = CommandConfig(dictation_prefixes=["скажи"])
        parser = CommandParser(config)

        action = parser.parse("скажи ")

        assert action is not None
        assert action.action_type == ActionType.TEXT
        assert action.value == "скажи"


class TestReplacements:
    def test_apply_replacements_basic(self) -> None:
        config = CommandConfig(replacements={"гитхаб": "GitHub", "питон": "Python"})
        parser = CommandParser(config)

        assert parser.apply_replacements("открой гитхаб") == "открой GitHub"

    def test_apply_replacements_case_insensitive(self) -> None:
        config = CommandConfig(replacements={"гитхаб": "GitHub"})
        parser = CommandParser(config)

        assert parser.apply_replacements("Открой Гитхаб") == "Открой GitHub"

    def test_apply_replacements_multiple(self) -> None:
        config = CommandConfig(replacements={"питон": "Python", "докер": "docker"})
        parser = CommandParser(config)

        assert parser.apply_replacements("запусти питон в докер") == "запусти Python в docker"

    def test_apply_replacements_word_boundary(self) -> None:
        config = CommandConfig(replacements={"баш": "bash"})
        parser = CommandParser(config)

        result = parser.apply_replacements("открой баш")
        assert result == "открой bash"

    def test_apply_replacements_no_match(self) -> None:
        config = CommandConfig(replacements={"гитхаб": "GitHub"})
        parser = CommandParser(config)

        assert parser.apply_replacements("привет мир") == "привет мир"

    def test_apply_replacements_empty(self) -> None:
        config = CommandConfig()
        parser = CommandParser(config)

        assert parser.apply_replacements("привет мир") == "привет мир"

    def test_load_replacements_from_separate_file(self) -> None:
        commands_content = """
control_commands: {}
"""
        replacements_content = """
гитхаб: GitHub
питон: Python
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as cf, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as rf:
            cf.write(commands_content)
            cf.flush()
            rf.write(replacements_content)
            rf.flush()
            config = CommandConfigLoader.load(Path(cf.name), Path(rf.name))

        assert config.replacements == {"гитхаб": "GitHub", "питон": "Python"}


class TestCommandConfig:
    def test_default_values(self) -> None:
        config = CommandConfig()

        assert config.control_commands == {}
        assert config.stop_commands == set()
        assert config.slash_commands == {}
        assert config.dictation_prefixes == []
        assert config.help_commands == set()
        assert config.replacements == {}


class TestCommandConfigLoader:
    def test_load_from_yaml_file(self) -> None:
        yaml_content = """
control_commands:
  клод:
    type: text
    value: claude
  таб:
    type: key
    value: tab
stop_commands:
  - стоп
  - хватит
slash_commands:
  хелп: /help
dictation_prefixes:
  - скажи
help_commands:
  - команды
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = CommandConfigLoader.load(Path(f.name))

        assert "клод" in config.control_commands
        assert config.control_commands["клод"].action_type == ActionType.TEXT
        assert config.control_commands["клод"].value == "claude"
        assert "таб" in config.control_commands
        assert config.stop_commands == {"стоп", "хватит"}
        assert config.slash_commands == {"хелп": "/help"}
        assert config.dictation_prefixes == ["скажи"]
        assert config.help_commands == {"команды"}

    def test_load_with_hotkey_action(self) -> None:
        yaml_content = """
control_commands:
  шифт таб:
    type: hotkey
    value:
      - shift
      - tab
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = CommandConfigLoader.load(Path(f.name))

        assert "шифт таб" in config.control_commands
        assert config.control_commands["шифт таб"].action_type == ActionType.HOTKEY
        assert config.control_commands["шифт таб"].value == ["shift", "tab"]

    def test_load_ignores_unknown_action_type(self) -> None:
        yaml_content = """
control_commands:
  unknown:
    type: unknown_type
    value: test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = CommandConfigLoader.load(Path(f.name))

        assert "unknown" not in config.control_commands

    def test_load_empty_config(self) -> None:
        yaml_content = "{}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = CommandConfigLoader.load(Path(f.name))

        assert config.control_commands == {}
        assert config.stop_commands == set()


class TestCommandParserFromFile:
    def test_from_config_file(self) -> None:
        yaml_content = """
control_commands:
  тест:
    type: text
    value: test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            parser = CommandParser.from_config_file(Path(f.name))

        action = parser.parse("тест")
        assert action is not None
        assert action.value == "test"
