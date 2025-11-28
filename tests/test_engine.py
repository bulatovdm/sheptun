# pyright: reportPrivateUsage=false
from unittest.mock import patch

import pytest

from sheptun.commands import CommandConfig, CommandParser
from sheptun.engine import BaseVoiceEngine
from sheptun.types import Action, ActionType, RecognitionResult


class MockStatusIndicator:
    def __init__(self) -> None:
        self.start_called = False
        self.stop_called = False
        self.listening_called = False
        self.processing_called = False
        self.idle_called = False
        self.last_error: str | None = None
        self.last_recognized: str | None = None
        self.last_action: str | None = None
        self.help_called = False

    def start(self) -> None:
        self.start_called = True

    def stop(self) -> None:
        self.stop_called = True

    def listening(self) -> None:
        self.listening_called = True

    def processing(self) -> None:
        self.processing_called = True

    def idle(self) -> None:
        self.idle_called = True

    def error(self, message: str) -> None:
        self.last_error = message

    def show_recognized(self, text: str) -> None:
        self.last_recognized = text

    def show_action(self, action_description: str) -> None:
        self.last_action = action_description

    def show_help(self) -> None:
        self.help_called = True


class MockRecognizer:
    def __init__(self, result: RecognitionResult | None = None) -> None:
        self._result = result

    def recognize(
        self, _audio_data: bytes, _sample_rate: int
    ) -> RecognitionResult | None:
        return self._result

    def start_warmup(self) -> None:
        pass

    def stop_warmup(self) -> None:
        pass


class MockKeyboardSender:
    def __init__(self) -> None:
        self.sent_text: list[str] = []
        self.sent_keys: list[str] = []
        self.sent_hotkeys: list[list[str]] = []

    def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    def send_key(self, key: str) -> None:
        self.sent_keys.append(key)

    def send_hotkey(self, keys: list[str]) -> None:
        self.sent_hotkeys.append(keys)


@pytest.fixture
def command_config() -> CommandConfig:
    return CommandConfig(
        control_commands={
            "клод": Action(ActionType.TEXT, "claude"),
            "таб": Action(ActionType.KEY, "tab"),
        },
        stop_commands={"стоп"},
        slash_commands={"хелп": "/help"},
        help_commands={"команды"},
    )


@pytest.fixture
def command_parser(command_config: CommandConfig) -> CommandParser:
    return CommandParser(command_config)


class TestBaseVoiceEngine:
    def test_start_sets_running(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()

            assert engine.is_running()
            assert status.listening_called

    def test_stop_clears_running(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()
            engine.stop()

            assert not engine.is_running()
            assert status.idle_called

    def test_execute_text_action(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            action = Action(ActionType.TEXT, "hello")
            engine._execute_action(action)

            assert "hello" in keyboard.sent_text

    def test_execute_key_action(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            action = Action(ActionType.KEY, "tab")
            engine._execute_action(action)

            assert "tab" in keyboard.sent_keys

    def test_execute_hotkey_action(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            action = Action(ActionType.HOTKEY, ["ctrl", "c"])
            engine._execute_action(action)

            assert ["ctrl", "c"] in keyboard.sent_hotkeys

    def test_execute_slash_action(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            action = Action(ActionType.SLASH, "/help")
            engine._execute_action(action)

            assert "/help" in keyboard.sent_text
            assert "return" in keyboard.sent_keys

    def test_execute_help_action(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            action = Action(ActionType.HELP, "")
            engine._execute_action(action)

            assert status.help_called

    def test_execute_stop_action(self, command_parser: CommandParser) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()
            action = Action(ActionType.STOP, "")
            engine._execute_action(action)

            assert not engine.is_running()
