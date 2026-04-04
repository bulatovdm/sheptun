import logging
import queue
import threading
from pathlib import Path

from sheptun.audio import AudioConfig, ContinuousAudioRecorder, VoiceActivityConfig
from sheptun.commands import CommandParser
from sheptun.dataset import DatasetRecorder
from sheptun.keyboard import MacOSKeyboardSender
from sheptun.recognition import WhisperRecognizer
from sheptun.settings import settings
from sheptun.status import ConsoleStatusIndicator, SimpleStatusIndicator
from sheptun.types import (
    Action,
    ActionType,
    AppState,
    KeyboardSender,
    SpeechRecognizer,
    StatusIndicator,
)

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
        record_dataset: bool = False,
    ) -> None:
        self._recognizer = recognizer
        self._command_parser = command_parser
        self._keyboard = keyboard_sender
        self._status = status_indicator
        self._audio_config = audio_config
        self._vad_config = vad_config
        self._recorder: ContinuousAudioRecorder | None = None
        self._dataset_recorder = DatasetRecorder() if record_dataset else None
        self._state = AppState.IDLE
        self._lock = threading.Lock()
        self._current_window_id: str | None = None
        self._window_text_sent: dict[str, bool] = {}
        self._recognition_queue: queue.Queue[bytes | None] = queue.Queue()
        self._recognition_thread: threading.Thread | None = None

    @property
    def state(self) -> AppState:
        with self._lock:
            return self._state

    @property
    def sample_rate(self) -> int:
        if self._recorder is not None:
            return self._recorder.sample_rate
        config = self._audio_config or AudioConfig()
        return config.sample_rate

    def _set_state(self, state: AppState) -> None:
        with self._lock:
            self._state = state

    def start(self) -> None:
        with self._lock:
            if self._state != AppState.IDLE:
                return
            self._state = AppState.RECORDING_TOGGLE
            self._window_text_sent.clear()

        self._on_start()
        self._recognizer.start_warmup()
        self._recognition_queue = queue.Queue()
        self._recognition_thread = threading.Thread(
            target=self._recognition_worker, daemon=True
        )
        self._recognition_thread.start()
        self._recorder = ContinuousAudioRecorder(self._audio_config, self._vad_config)
        self._recorder.set_speech_start_callback(self._on_speech_started)
        self._recorder.set_callback(self._on_speech_detected)
        self._recorder.start()

    def stop(self) -> None:
        with self._lock:
            if self._state == AppState.IDLE:
                return

        self._recognizer.stop_warmup()
        if self._recorder is not None:
            self._recorder.stop()
        self._recognition_queue.put(None)
        if self._recognition_thread is not None:
            self._recognition_thread.join(timeout=10.0)
            self._recognition_thread = None
        self._recorder = None

        with self._lock:
            self._state = AppState.IDLE
        self._on_stop()

    def is_running(self) -> bool:
        with self._lock:
            return self._state != AppState.IDLE

    def set_keyboard_sender(self, keyboard_sender: KeyboardSender) -> None:
        """Set the keyboard sender to use for text input."""
        self._keyboard = keyboard_sender

    def recognize_and_execute(self, audio_data: bytes) -> None:
        try:
            result = self._recognizer.recognize(audio_data, self.sample_rate)
            if result and result.text:
                text = self._command_parser.apply_replacements(result.text)
                self._log(f"Recognized: '{text}'")
                self._save_to_dataset(audio_data, result.text)
                action = self._command_parser.parse(text)
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

    def _save_to_dataset(
        self, audio_data: bytes, text: str, original_text: str | None = None
    ) -> None:
        if self._dataset_recorder is None:
            return

        import numpy as np

        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        # Save original text as main, corrected as optional field
        save_text = original_text if original_text else text
        corrected = text if original_text else None
        path = self._dataset_recorder.save(audio_float, save_text, corrected)
        self._log(f"Saved to dataset: {path}")
        if original_text:
            self._log(f"Original: '{original_text}' -> Corrected: '{text}'")

    def _on_speech_started(self) -> None:
        if self.state == AppState.IDLE:
            return
        self._log("Speech started, capturing focus")
        self._keyboard.start_capture()
        self._update_current_window()

    def _on_speech_detected(self, audio_data: bytes) -> None:
        if self.state == AppState.IDLE:
            return
        self._log(f"Speech detected: {len(audio_data)} bytes")
        self._recognition_queue.put(audio_data)

    def _recognition_worker(self) -> None:
        while True:
            audio_data = self._recognition_queue.get()
            if audio_data is None:
                break
            self._process_speech(audio_data)

    def _process_speech(self, audio_data: bytes) -> None:
        if self.state == AppState.IDLE:
            return

        self._set_state(AppState.PROCESSING)
        self._status.processing()

        try:
            result = self._recognizer.recognize(audio_data, self.sample_rate)

            if result is None or not result.text:
                self._log("Recognition returned empty result")
                self._keyboard.end_capture()
                self._resume_listening()
                return

            text = self._command_parser.apply_replacements(result.text)
            if text != result.text:
                self._log(f"Recognized: '{result.text}' -> '{text}'")
            else:
                self._log(f"Recognized: '{text}'")
            self._status.show_recognized(text)
            self._save_to_dataset(audio_data, result.text, result.original_text)
            action = self._command_parser.parse(text)
            self._log(f"Parsed action: {action}")

            if action is not None:
                self._execute_action(action)

        except Exception as e:
            self._log(f"Error processing speech: {e}")
            self._status.error(str(e))

        self._keyboard.end_capture()
        self._resume_listening()

    def _resume_listening(self) -> None:
        if self.state != AppState.IDLE:
            self._set_state(AppState.RECORDING_TOGGLE)
            self._status.listening()

    def _execute_action(self, action: Action) -> None:
        match action.action_type:
            case ActionType.STOP:
                self._status.show_action("Остановка")
                self._handle_stop()

            case ActionType.TEXT:
                if isinstance(action.value, str):
                    text = self._prepare_text(action.value)
                    self._status.show_action(f"Ввод текста: {text}")
                    self._keyboard.send_text(text)
                    if self._current_window_id:
                        self._window_text_sent[self._current_window_id] = True

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

    def _prepare_text(self, text: str) -> str:
        if not settings.auto_space or not text or not text[0].isalpha():
            return text
        cursor_position = self._keyboard.get_cursor_position()
        if cursor_position > 0:
            return " " + text
        if cursor_position == 0 and self._current_window_id:
            self._window_text_sent[self._current_window_id] = False
        text_sent = self._window_text_sent.get(self._current_window_id or "", False)
        if text_sent:
            return " " + text
        return text

    def _update_current_window(self) -> None:
        new_window = self._get_current_window_id()
        if new_window != self._current_window_id:
            self._log(f"Window changed: {self._current_window_id} -> {new_window}")
        self._current_window_id = new_window

    def _get_current_window_id(self) -> str | None:
        try:
            from sheptun.focus import FocusTracker

            tracker = FocusTracker()
            state = tracker.get_current_state()
            if state.app_bundle_id:
                return f"{state.app_bundle_id}:{state.window_title or ''}"
            return None
        except Exception:
            return None


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
        record_dataset: bool = False,
    ) -> None:
        super().__init__(
            recognizer=recognizer,
            command_parser=command_parser,
            keyboard_sender=keyboard_sender,
            status_indicator=status_indicator,
            audio_config=audio_config,
            vad_config=vad_config,
            record_dataset=record_dataset,
        )
        self._debug = debug

    @classmethod
    def create(
        cls,
        config_path: Path,
        model_name: str = "base",
        device: str | None = None,
        use_live_status: bool = True,
        debug: bool = False,
        replacements_path: Path | None = None,
    ) -> "VoiceEngine":
        recognizer = cls._create_recognizer(model_name, device)
        command_parser = CommandParser.from_config_file(config_path, replacements_path)
        keyboard_sender = MacOSKeyboardSender(use_clipboard=settings.use_clipboard)
        status_indicator: ConsoleStatusIndicator | SimpleStatusIndicator = (
            ConsoleStatusIndicator() if use_live_status else SimpleStatusIndicator()
        )

        return cls(
            recognizer=recognizer,
            command_parser=command_parser,
            keyboard_sender=keyboard_sender,
            status_indicator=status_indicator,
            debug=debug,
            record_dataset=settings.record_dataset,
        )

    @staticmethod
    def _create_recognizer(model_name: str, device: str | None) -> SpeechRecognizer:
        if settings.recognizer == "apple":
            from sheptun.apple_speech import AppleSpeechRecognizer

            logger.info("Using Apple Speech Framework for recognition")
            return AppleSpeechRecognizer()

        if settings.recognizer == "mlx":
            from sheptun.recognition import MLXWhisperRecognizer

            logger.info(f"Using MLX Whisper ({model_name}) for recognition")
            return MLXWhisperRecognizer(model_name=model_name)

        if settings.recognizer == "parakeet":
            from sheptun.parakeet import ParakeetRecognizer

            logger.info("Using Parakeet TDT for recognition")
            return ParakeetRecognizer()

        if settings.recognizer == "qwen":
            from sheptun.qwen_asr import QwenASRRecognizer

            logger.info("Using Qwen3-ASR for recognition")
            return QwenASRRecognizer()

        from sheptun.recognition import is_local_model

        if is_local_model(model_name):
            from sheptun.recognition import HuggingFaceWhisperRecognizer

            logger.info(f"Using HuggingFace Whisper ({model_name}) for recognition")
            return HuggingFaceWhisperRecognizer(model_path=model_name, device=device)

        logger.info(f"Using Whisper ({model_name}) for recognition")
        return WhisperRecognizer(model_name=model_name, device=device)

    def _on_start(self) -> None:
        self._status.start()
        self._status.listening()

    def _on_stop(self) -> None:
        self._status.stop()

    def _log(self, message: str) -> None:
        if self._debug:
            logger.debug(message)
