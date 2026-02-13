# pyright: reportPrivateUsage=false
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    def __init__(
        self,
        result: RecognitionResult | None = None,
        results: "Sequence[RecognitionResult | None] | None" = None,
        delay: float = 0.0,
    ) -> None:
        self._result = result
        self._results = list(results) if results else None
        self._delay = delay
        self._call_count = 0

    def recognize(self, _audio_data: bytes, _sample_rate: int) -> RecognitionResult | None:
        if self._delay > 0:
            time.sleep(self._delay)
        self._call_count += 1
        if self._results:
            return self._results.pop(0) if self._results else self._result
        return self._result

    def start_warmup(self) -> None:
        pass

    def stop_warmup(self) -> None:
        pass


class MockKeyboardSender:
    def __init__(self, cursor_position: int = -1) -> None:
        self.sent_text: list[str] = []
        self.sent_keys: list[str] = []
        self.sent_hotkeys: list[list[str]] = []
        self._cursor_position = cursor_position

    def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    def send_key(self, key: str) -> None:
        self.sent_keys.append(key)

    def send_hotkey(self, keys: list[str]) -> None:
        self.sent_hotkeys.append(keys)

    def start_capture(self) -> None:
        pass

    def end_capture(self) -> None:
        pass

    def has_text_before_cursor(self) -> bool:
        return self._cursor_position > 0

    def get_cursor_position(self) -> int:
        return self._cursor_position


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

    def test_execute_text_action_first_in_session(self, command_parser: CommandParser) -> None:
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

    def test_execute_text_action_subsequent_adds_space(self, command_parser: CommandParser) -> None:
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
            engine._current_window_id = "test-window"
            action = Action(ActionType.TEXT, "hello")
            engine._execute_action(action)
            engine._execute_action(action)
            assert " hello" in keyboard.sent_text

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

    def test_speech_detected_queues_for_recognition(
        self, command_parser: CommandParser
    ) -> None:
        result = RecognitionResult(text="привет", confidence=0.9)
        recognizer = MockRecognizer(result=result)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()

            engine._on_speech_detected(b"\x00" * 1000)
            assert not engine._recognition_queue.empty()

            engine.stop()

    def test_recognition_worker_processes_speech(
        self, command_parser: CommandParser
    ) -> None:
        result = RecognitionResult(text="привет", confidence=0.9)
        recognizer = MockRecognizer(result=result)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()

            engine._on_speech_detected(b"\x00" * 1000)
            engine.stop()

            assert "привет" in keyboard.sent_text

    def test_multiple_phrases_processed_sequentially(
        self, command_parser: CommandParser
    ) -> None:
        results = [
            RecognitionResult(text="раз", confidence=0.9),
            RecognitionResult(text="два", confidence=0.9),
            RecognitionResult(text="три", confidence=0.9),
        ]
        recognizer = MockRecognizer(results=results, delay=0.05)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()

            engine._on_speech_detected(b"\x00" * 1000)
            engine._on_speech_detected(b"\x00" * 2000)
            engine._on_speech_detected(b"\x00" * 3000)

            engine.stop()

            assert "раз" in keyboard.sent_text
            assert "два" in keyboard.sent_text
            assert "три" in keyboard.sent_text
            assert recognizer._call_count == 3

    def test_speech_detected_ignored_when_idle(
        self, command_parser: CommandParser
    ) -> None:
        result = RecognitionResult(text="привет", confidence=0.9)
        recognizer = MockRecognizer(result=result)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine._on_speech_detected(b"\x00" * 1000)

            assert engine._recognition_queue.empty()
            assert len(keyboard.sent_text) == 0

    def test_stop_waits_for_pending_recognition(
        self, command_parser: CommandParser
    ) -> None:
        result = RecognitionResult(text="тест", confidence=0.9)
        recognizer = MockRecognizer(result=result, delay=0.1)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()

            engine._on_speech_detected(b"\x00" * 1000)
            time.sleep(0.01)
            engine.stop()

            assert "тест" in keyboard.sent_text

    def test_callback_thread_not_blocked_during_recognition(
        self, command_parser: CommandParser
    ) -> None:
        result = RecognitionResult(text="привет", confidence=0.9)
        recognizer = MockRecognizer(result=result, delay=0.2)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()

            engine._on_speech_detected(b"\x00" * 1000)
            start = time.monotonic()
            engine._on_speech_detected(b"\x00" * 2000)
            elapsed = time.monotonic() - start

            assert elapsed < 0.1

            engine.stop()

    def test_engine_works_after_stop_and_restart(
        self, command_parser: CommandParser
    ) -> None:
        results = [
            RecognitionResult(text="до", confidence=0.9),
            RecognitionResult(text="после", confidence=0.9),
        ]
        recognizer = MockRecognizer(results=results)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()
            engine._on_speech_detected(b"\x00" * 1000)
            engine.stop()

            assert "до" in keyboard.sent_text

            engine.start()
            engine._on_speech_detected(b"\x00" * 2000)
            engine.stop()

            assert "после" in keyboard.sent_text

    def test_start_creates_fresh_recorder(
        self, command_parser: CommandParser
    ) -> None:
        status = MockStatusIndicator()
        recognizer = MockRecognizer()
        keyboard = MockKeyboardSender()

        with patch("sheptun.engine.ContinuousAudioRecorder") as MockRecorder:
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()
            engine.stop()

            engine.start()
            engine.stop()

            assert MockRecorder.call_count == 2

    def test_start_creates_fresh_recognition_queue(
        self, command_parser: CommandParser
    ) -> None:
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
            first_queue = engine._recognition_queue
            engine.stop()

            engine.start()
            second_queue = engine._recognition_queue
            engine.stop()

            assert first_queue is not second_queue

    def test_no_stale_events_across_toggle_cycles(
        self, command_parser: CommandParser
    ) -> None:
        results = [
            RecognitionResult(text="сессия1", confidence=0.9),
            RecognitionResult(text="сессия2", confidence=0.9),
        ]
        recognizer = MockRecognizer(results=results)
        keyboard = MockKeyboardSender()
        status = MockStatusIndicator()

        with patch("sheptun.engine.ContinuousAudioRecorder"):
            engine = BaseVoiceEngine(
                recognizer=recognizer,  # type: ignore[arg-type]
                command_parser=command_parser,
                keyboard_sender=keyboard,  # type: ignore[arg-type]
                status_indicator=status,  # type: ignore[arg-type]
            )
            engine.start()
            engine._on_speech_detected(b"\x00" * 1000)
            engine.stop()

            assert "сессия1" in keyboard.sent_text

            engine.start()
            engine._on_speech_detected(b"\x00" * 2000)
            engine.stop()

            assert "сессия2" in keyboard.sent_text
            assert keyboard.sent_text.count("сессия1") == 1
