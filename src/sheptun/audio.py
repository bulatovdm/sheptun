import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

logger = logging.getLogger("sheptun.audio")

DEFAULT_BLOCKSIZE = 1024


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = DEFAULT_BLOCKSIZE


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
        idle_timeout=settings.idle_timeout,
    )


@dataclass
class VoiceActivityConfig:
    energy_threshold: float = 0.01
    silence_duration: float = 0.5
    min_speech_duration: float = 0.2
    max_speech_duration: float = 30.0
    idle_timeout: float = 5.0  # Reset buffer if no speech detected for this duration


class EnergyVAD:
    """Energy-based Voice Activity Detector."""

    def __init__(self, config: VoiceActivityConfig) -> None:
        self.config = config
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


class SileroVAD:
    """Silero neural network Voice Activity Detector."""

    def __init__(self, config: VoiceActivityConfig) -> None:
        from silero_vad import load_silero_vad  # type: ignore[import-untyped]

        self.config = config
        self._model = load_silero_vad()  # type: ignore[no-untyped-call]
        self._silence_samples = 0
        self._speech_samples = 0
        self._is_speaking = False

    def reset(self) -> None:
        self._model.reset_states()  # type: ignore[union-attr]
        self._silence_samples = 0
        self._speech_samples = 0
        self._is_speaking = False

    def process_chunk(self, audio_chunk: bytes, sample_rate: int) -> bool:
        import torch  # type: ignore[import-untyped]

        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_array)  # type: ignore[no-untyped-call]

        speech_prob: float = self._model(audio_tensor, sample_rate).item()  # type: ignore[union-attr]

        if speech_prob > 0.5:
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


SILERO_BLOCKSIZE = 512  # Silero VAD requires exactly 512 samples for 16kHz


def get_vad_blocksize(vad_type: str) -> int:
    if vad_type == "silero":
        return SILERO_BLOCKSIZE
    return DEFAULT_BLOCKSIZE


def create_vad(config: VoiceActivityConfig, vad_type: str = "energy") -> EnergyVAD | SileroVAD:
    if vad_type == "silero":
        return SileroVAD(config)
    return EnergyVAD(config)


class ContinuousAudioRecorder:
    def __init__(
        self,
        audio_config: AudioConfig | None = None,
        vad_config: VoiceActivityConfig | None = None,
        vad_type: str | None = None,
    ) -> None:
        from sheptun.settings import settings

        vad_type_to_use = vad_type if vad_type is not None else settings.vad_type

        if audio_config is None:
            audio_config = AudioConfig(blocksize=get_vad_blocksize(vad_type_to_use))

        self._audio_config = audio_config
        self._vad_config = vad_config or _default_vad_config()
        self._vad = create_vad(self._vad_config, vad_type=vad_type_to_use)
        self._buffer: list[bytes] = []
        self._stream: sd.InputStream | None = None
        self._running = False
        self._lock = threading.Lock()
        self._on_speech_start: Callable[[], None] | None = None
        self._on_speech_complete: Callable[[bytes], None] | None = None
        self._last_speech_time: float = 0.0
        self._speech_started_notified: bool = False

    @property
    def sample_rate(self) -> int:
        return self._audio_config.sample_rate

    def set_callback(self, callback: Callable[[bytes], None]) -> None:
        self._on_speech_complete = callback

    def set_speech_start_callback(self, callback: Callable[[], None]) -> None:
        self._on_speech_start = callback

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._vad.reset()
        self._last_speech_time = time.time()
        self._speech_started_notified = False

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
        current_time = time.time()
        notify_speech_start = False

        with self._lock:
            self._buffer.append(chunk)
            was_speaking = self._vad._is_speaking
            speech_complete = self._vad.process_chunk(chunk, self._audio_config.sample_rate)

            if self._vad._is_speaking:
                self._last_speech_time = current_time
                if not was_speaking and not self._speech_started_notified:
                    notify_speech_start = True
                    self._speech_started_notified = True

            if speech_complete:
                audio_data = b"".join(self._buffer)
                self._buffer.clear()
                self._vad.reset()
                self._last_speech_time = current_time
                self._speech_started_notified = False
            elif self._should_reset_idle(current_time) or self._is_buffer_too_large():
                buffer_duration = len(self._buffer) * self._audio_config.blocksize / self._audio_config.sample_rate
                if buffer_duration > 0.5:
                    logger.debug(f"Buffer reset: {buffer_duration:.1f}s of buffered audio")
                self._buffer.clear()
                self._vad.reset()
                self._last_speech_time = current_time
                self._speech_started_notified = False

        if notify_speech_start and self._on_speech_start is not None:
            self._on_speech_start()

        if audio_data is not None and self._on_speech_complete is not None:
            self._on_speech_complete(audio_data)

    def _should_reset_idle(self, current_time: float) -> bool:
        if self._vad._is_speaking:
            return False
        idle_duration = current_time - self._last_speech_time
        return idle_duration >= self._vad_config.idle_timeout

    def _is_buffer_too_large(self) -> bool:
        buffer_duration = len(self._buffer) * self._audio_config.blocksize / self._audio_config.sample_rate
        return buffer_duration > self._vad_config.max_speech_duration * 2
