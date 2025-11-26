from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol


class ActionType(Enum):
    TEXT = auto()
    KEY = auto()
    HOTKEY = auto()
    STOP = auto()
    SLASH = auto()
    HELP = auto()


@dataclass(frozen=True, slots=True)
class Action:
    action_type: ActionType
    value: str | list[str]


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    text: str
    confidence: float


class SpeechRecognizer(Protocol):
    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None: ...


class KeyboardSender(Protocol):
    def send_text(self, text: str) -> None: ...
    def send_key(self, key: str) -> None: ...
    def send_hotkey(self, keys: list[str]) -> None: ...


class AudioRecorder(Protocol):
    def start(self) -> None: ...
    def stop(self) -> bytes: ...
    def is_recording(self) -> bool: ...


class StatusIndicator(Protocol):
    def listening(self) -> None: ...
    def processing(self) -> None: ...
    def error(self, message: str) -> None: ...
    def idle(self) -> None: ...
