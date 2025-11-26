import threading
from pathlib import Path

from sheptun.audio import AudioConfig, ContinuousAudioRecorder, VoiceActivityConfig
from sheptun.commands import CommandParser
from sheptun.keyboard import MacOSKeyboardSender
from sheptun.recognition import WhisperRecognizer
from sheptun.status import ConsoleStatusIndicator, SimpleStatusIndicator
from sheptun.types import Action, ActionType


class VoiceEngine:
    def __init__(
        self,
        recognizer: WhisperRecognizer,
        command_parser: CommandParser,
        keyboard_sender: MacOSKeyboardSender,
        status_indicator: ConsoleStatusIndicator | SimpleStatusIndicator,
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

    @classmethod
    def create(
        cls,
        config_path: Path,
        model_name: str = "base",
        device: str | None = None,
        use_live_status: bool = True,
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
        )

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        self._status.start()
        self._status.listening()
        self._recorder.set_callback(self._on_speech_detected)
        self._recorder.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._recorder.stop()
        self._status.stop()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _on_speech_detected(self, audio_data: bytes) -> None:
        if not self._running:
            return

        self._status.processing()

        try:
            result = self._recognizer.recognize(audio_data, self._recorder.sample_rate)

            if result is None or not result.text:
                self._status.listening()
                return

            self._status.show_recognized(result.text)
            action = self._command_parser.parse(result.text)

            if action is not None:
                self._execute_action(action)

        except Exception as e:
            self._status.error(str(e))

        if self._running:
            self._status.listening()

    def _execute_action(self, action: Action) -> None:
        match action.action_type:
            case ActionType.STOP:
                self._status.show_action("Остановка")
                self.stop()

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
