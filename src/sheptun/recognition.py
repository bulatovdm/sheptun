import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import whisper

from sheptun.settings import settings
from sheptun.types import RecognitionResult

logger = logging.getLogger("sheptun")

# Pattern for sound descriptions in Cyrillic caps (e.g., laughter, music)
_SOUND_DESCRIPTION_PATTERN = re.compile(r"^[А-ЯЁ\s]+$")

# Short silence for warmup (0.1 sec at 16kHz)
_WARMUP_AUDIO = np.zeros(1600, dtype=np.float32)


class WhisperRecognizer:
    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        self._model: Any = whisper.load_model(model_name, device=device)
        self._model_name = model_name
        self._hallucinations = {
            h.lower() for h in (hallucinations or settings.hallucinations)
        }
        self._warmup_interval = (
            warmup_interval if warmup_interval is not None else settings.warmup_interval
        )
        self._warmup_timer: threading.Timer | None = None
        self._warmup_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def warmup(self) -> None:
        """Run a quick inference on silence to keep the model in GPU memory."""
        try:
            self._model.transcribe(_WARMUP_AUDIO, language="ru", fp16=False)
            logger.debug("Warmup completed")
        except Exception as e:
            logger.debug(f"Warmup error: {e}")

    def start_warmup(self) -> None:
        """Start periodic warmup to keep model in memory."""
        if self._warmup_interval <= 0:
            return

        self._schedule_warmup()
        logger.debug(f"Warmup started with interval {self._warmup_interval}s")

    def stop_warmup(self) -> None:
        """Stop periodic warmup."""
        with self._warmup_lock:
            if self._warmup_timer is not None:
                self._warmup_timer.cancel()
                self._warmup_timer = None
        logger.debug("Warmup stopped")

    def _schedule_warmup(self) -> None:
        with self._warmup_lock:
            if self._warmup_timer is not None:
                self._warmup_timer.cancel()

            self._warmup_timer = threading.Timer(self._warmup_interval, self._warmup_tick)
            self._warmup_timer.daemon = True
            self._warmup_timer.start()

    def _warmup_tick(self) -> None:
        self.warmup()
        self._schedule_warmup()

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        audio_array = self._bytes_to_float_array(audio_data, sample_rate)
        if audio_array is None:
            return None

        result = self._model.transcribe(
            audio_array,
            language="ru",
            fp16=False,
            task="transcribe",
        )

        text = result.get("text", "").strip()
        if not text:
            return None

        if self._is_hallucination(text):
            return None

        segments = result.get("segments", [])
        confidence = self._calculate_confidence(segments)

        return RecognitionResult(text=text, confidence=confidence)

    def recognize_from_file(self, audio_path: Path) -> RecognitionResult | None:
        result = self._model.transcribe(
            str(audio_path),
            language="ru",
            fp16=False,
            task="transcribe",
        )

        text = result.get("text", "").strip()
        if not text:
            return None

        if self._is_hallucination(text):
            return None

        segments = result.get("segments", [])
        confidence = self._calculate_confidence(segments)

        return RecognitionResult(text=text, confidence=confidence)

    def _is_hallucination(self, text: str) -> bool:
        text_lower = text.lower()
        if any(h in text_lower for h in self._hallucinations):
            return True
        # Filter sound descriptions in Cyrillic caps (e.g., laughter, music)
        return bool(_SOUND_DESCRIPTION_PATTERN.match(text.strip()))

    def _bytes_to_float_array(
        self, audio_data: bytes, sample_rate: int
    ) -> np.ndarray[Any, Any] | None:
        if len(audio_data) == 0:
            return None

        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if sample_rate != 16000:
            audio_float32 = self._resample(audio_float32, sample_rate, 16000)

        audio_float32 = self._trim_silence(audio_float32, sample_rate)

        return audio_float32

    def _trim_silence(
        self,
        audio: np.ndarray[Any, Any],
        sample_rate: int,
        threshold: float = 0.01,
        window_ms: int = 20,
    ) -> np.ndarray[Any, Any]:
        window_size = int(sample_rate * window_ms / 1000)
        if len(audio) < window_size:
            return audio

        start = self._find_speech_start(audio, window_size, threshold, sample_rate)
        end = self._find_speech_end(audio, window_size, threshold, sample_rate)

        trimmed = audio[start:end]
        trimmed_duration = (len(audio) - len(trimmed)) / sample_rate
        if trimmed_duration > 0.1:
            logger.debug(f"Trimmed {trimmed_duration:.2f}s silence")

        return trimmed

    def _find_speech_start(
        self,
        audio: np.ndarray[Any, Any],
        window_size: int,
        threshold: float,
        sample_rate: int,
    ) -> int:
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i : i + window_size]
            energy = np.sqrt(np.mean(window**2))
            if energy > threshold:
                return max(0, i - int(sample_rate * 0.05))
        return 0

    def _find_speech_end(
        self,
        audio: np.ndarray[Any, Any],
        window_size: int,
        threshold: float,
        sample_rate: int,
    ) -> int:
        for i in range(len(audio) - window_size, 0, -window_size):
            window = audio[i : i + window_size]
            energy = np.sqrt(np.mean(window**2))
            if energy > threshold:
                return min(len(audio), i + window_size + int(sample_rate * 0.05))
        return len(audio)

    def _resample(
        self, audio: np.ndarray[Any, Any], orig_sr: int, target_sr: int
    ) -> np.ndarray[Any, Any]:
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled: np.ndarray[Any, Any] = np.interp(indices, np.arange(len(audio)), audio).astype(
            np.float32
        )
        return resampled

    def _calculate_confidence(self, segments: list[dict[str, Any]]) -> float:
        if not segments:
            return 0.0

        total_prob = 0.0
        total_tokens = 0

        for segment in segments:
            avg_logprob = segment.get("avg_logprob", -1.0)
            no_speech_prob = segment.get("no_speech_prob", 0.0)

            if no_speech_prob > 0.5:
                continue

            prob = np.exp(avg_logprob)
            tokens = len(segment.get("tokens", [1]))
            total_prob += prob * tokens
            total_tokens += tokens

        if total_tokens == 0:
            return 0.0

        return float(total_prob / total_tokens)
