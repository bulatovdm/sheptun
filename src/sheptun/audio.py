import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = 1024


class AudioRecorder:
    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
        self._buffer: list[bytes] = []
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    def start(self) -> None:
        if self._recording:
            return

        with self._lock:
            self._buffer.clear()
            self._recording = True

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype=self._config.dtype,
            blocksize=self._config.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        if not self._recording:
            return b""

        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            audio_data = b"".join(self._buffer)
            self._buffer.clear()

        return audio_data

    def is_recording(self) -> bool:
        return self._recording

    def _audio_callback(
        self,
        indata: np.ndarray[Any, np.dtype[np.int16]],
        _frames: int,
        _time_info: dict[str, float],
        _status: sd.CallbackFlags,
    ) -> None:
        if not self._recording:
            return

        with self._lock:
            self._buffer.append(indata.tobytes())


def _default_vad_config() -> "VoiceActivityConfig":
    from sheptun.settings import settings

    return VoiceActivityConfig(
        energy_threshold=settings.energy_threshold,
        silence_duration=settings.silence_duration,
        min_speech_duration=settings.min_speech_duration,
        max_speech_duration=settings.max_speech_duration,
    )


@dataclass
class VoiceActivityConfig:
    energy_threshold: float = 0.01
    silence_duration: float = 0.5
    min_speech_duration: float = 0.2
    max_speech_duration: float = 30.0


@dataclass
class VoiceActivityDetector:
    config: VoiceActivityConfig = field(default_factory=VoiceActivityConfig)
    _silence_samples: int = field(init=False, default=0)
    _speech_samples: int = field(init=False, default=0)
    _is_speaking: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._silence_samples = 0
        self._speech_samples = 0
        self._is_speaking = False

    def reset(self) -> None:
        self._silence_samples = 0
        self._speech_samples = 0
        self._is_speaking = False

    def process_chunk(self, audio_chunk: bytes, sample_rate: int) -> bool:
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(audio_array**2))

        if energy > self.config.energy_threshold:
            self._speech_samples += len(audio_array)
            self._silence_samples = 0
            self._is_speaking = True
        else:
            self._silence_samples += len(audio_array)

        speech_duration = self._speech_samples / sample_rate
        silence_duration = self._silence_samples / sample_rate

        if speech_duration >= self.config.max_speech_duration:
            return True

        if not self._is_speaking:
            return False

        if silence_duration >= self.config.silence_duration:
            return speech_duration >= self.config.min_speech_duration

        return False


class ContinuousAudioRecorder:
    def __init__(
        self,
        audio_config: AudioConfig | None = None,
        vad_config: VoiceActivityConfig | None = None,
    ) -> None:
        self._audio_config = audio_config or AudioConfig()
        self._vad = VoiceActivityDetector(vad_config or _default_vad_config())
        self._buffer: list[bytes] = []
        self._stream: sd.InputStream | None = None
        self._running = False
        self._lock = threading.Lock()
        self._on_speech_complete: Callable[[bytes], None] | None = None

    @property
    def sample_rate(self) -> int:
        return self._audio_config.sample_rate

    def set_callback(self, callback: Callable[[bytes], None]) -> None:
        self._on_speech_complete = callback

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._vad.reset()

        with self._lock:
            self._buffer.clear()

        self._stream = sd.InputStream(
            samplerate=self._audio_config.sample_rate,
            channels=self._audio_config.channels,
            dtype=self._audio_config.dtype,
            blocksize=self._audio_config.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def is_running(self) -> bool:
        return self._running

    def _audio_callback(
        self,
        indata: np.ndarray[Any, np.dtype[np.int16]],
        _frames: int,
        _time_info: dict[str, float],
        _status: sd.CallbackFlags,
    ) -> None:
        if not self._running:
            return

        chunk = indata.tobytes()
        audio_data: bytes | None = None

        with self._lock:
            self._buffer.append(chunk)
            speech_complete = self._vad.process_chunk(chunk, self._audio_config.sample_rate)

            if speech_complete:
                audio_data = b"".join(self._buffer)
                self._buffer.clear()
                self._vad.reset()

        if audio_data is not None and self._on_speech_complete is not None:
            self._on_speech_complete(audio_data)
