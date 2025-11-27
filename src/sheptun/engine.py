import logging
import threading
from pathlib import Path

from sheptun.audio import AudioConfig, ContinuousAudioRecorder, VoiceActivityConfig
from sheptun.commands import CommandParser
from sheptun.keyboard import MacOSKeyboardSender
from sheptun.recognition import WhisperRecognizer
from sheptun.status import ConsoleStatusIndicator, SimpleStatusIndicator
from sheptun.types import Action, ActionType, KeyboardSender, SpeechRecognizer, StatusIndicator

logger = logging.getLogger("sheptun")


class BaseVoiceEngine:
    def __init__(
        self,
        recognizer: SpeechRecognizer,
        command_parser: CommandParser,
        keyboard_sender: KeyboardSender,
        status_indicator: StatusIndicator,
        audio_config: AudioConfig | None = None,
        vad_config: VoiceActivityConfig | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._command_parser = command_parser
        self._keyboard = keyboard_sender
        self._status = status_indicator
        self._recorder = ContinuousAudioRecorder(audio_config, vad_config)
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        self._on_start()
        self._recorder.set_callback(self._on_speech_detected)
        self._recorder.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._recorder.stop()
        self._on_stop()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def recognize_and_execute(self, audio_data: bytes) -> None:
        try:
            result = self._recognizer.recognize(audio_data, self._recorder.sample_rate)
            if result and result.text:
                self._log(f"Recognized: '{result.text}'")
                action = self._command_parser.parse(result.text)
                if action:
                    self._execute_action(action)
        except Exception as e:
            self._log(f"Recognition error: {e}")

    def _on_start(self) -> None:
        self._status.listening()

    def _on_stop(self) -> None:
        self._status.idle()

    def _log(self, message: str) -> None:
        logger.debug(message)

    def _on_speech_detected(self, audio_data: bytes) -> None:
        if not self._running:
            return

        self._log(f"Speech detected: {len(audio_data)} bytes")
        self._status.processing()

        try:
            result = self._recognizer.recognize(audio_data, self._recorder.sample_rate)

            if result is None or not result.text:
                self._log("Recognition returned empty result")
                self._status.listening()
                return

            self._log(f"Recognized: '{result.text}'")
            self._status.show_recognized(result.text)
            action = self._command_parser.parse(result.text)
            self._log(f"Parsed action: {action}")

            if action is not None:
                self._execute_action(action)

        except Exception as e:
            self._log(f"Error processing speech: {e}")
            self._status.error(str(e))

        if self._running:
            self._status.listening()

    def _execute_action(self, action: Action) -> None:
        match action.action_type:
            case ActionType.STOP:
                self._status.show_action("Остановка")
                self._handle_stop()

            case ActionType.TEXT:
                if isinstance(action.value, str):
                    self._status.show_action(f"Ввод текста: {action.value}")
                    self._keyboard.send_text(action.value)

            case ActionType.KEY:
                if isinstance(action.value, str):
                    self._status.show_action(f"Клавиша: {action.value}")
                    self._keyboard.send_key(action.value)

            case ActionType.HOTKEY:
                if isinstance(action.value, list):
                    keys_str = "+".join(action.value)
                    self._status.show_action(f"Комбинация: {keys_str}")
                    self._keyboard.send_hotkey(action.value)

            case ActionType.SLASH:
                if isinstance(action.value, str):
                    self._status.show_action(f"Slash-команда: {action.value}")
                    self._keyboard.send_text(action.value)
                    self._keyboard.send_key("return")

            case ActionType.HELP:
                self._status.show_help()

    def _handle_stop(self) -> None:
        self.stop()


class VoiceEngine(BaseVoiceEngine):
    def __init__(
        self,
        recognizer: SpeechRecognizer,
        command_parser: CommandParser,
        keyboard_sender: KeyboardSender,
        status_indicator: StatusIndicator,
        audio_config: AudioConfig | None = None,
        vad_config: VoiceActivityConfig | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            recognizer=recognizer,
            command_parser=command_parser,
            keyboard_sender=keyboard_sender,
            status_indicator=status_indicator,
            audio_config=audio_config,
            vad_config=vad_config,
        )
        self._stop_requested = False
        self._debug = debug

    @classmethod
    def create(
        cls,
        config_path: Path,
        model_name: str = "base",
        device: str | None = None,
        use_live_status: bool = True,
        debug: bool = False,
    ) -> "VoiceEngine":
        recognizer = WhisperRecognizer(model_name=model_name, device=device)
        command_parser = CommandParser.from_config_file(config_path)
        keyboard_sender = MacOSKeyboardSender()
        status_indicator: ConsoleStatusIndicator | SimpleStatusIndicator = (
            ConsoleStatusIndicator() if use_live_status else SimpleStatusIndicator()
        )

        return cls(
            recognizer=recognizer,
            command_parser=command_parser,
            keyboard_sender=keyboard_sender,
            status_indicator=status_indicator,
            debug=debug,
        )

    def _on_start(self) -> None:
        self._stop_requested = False
        self._status.start()
        self._status.listening()

    def _on_stop(self) -> None:
        self._status.stop()

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def is_running(self) -> bool:
        with self._lock:
            if self._stop_requested and self._running:
                return False
            return self._running

    def _log(self, message: str) -> None:
        if self._debug:
            logger.debug(message)

    def _handle_stop(self) -> None:
        self.request_stop()
