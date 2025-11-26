from pathlib import Path
from typing import Any

import numpy as np
import whisper

from sheptun.types import RecognitionResult


class WhisperRecognizer:
    def __init__(self, model_name: str = "base", device: str | None = None) -> None:
        self._model: Any = whisper.load_model(model_name, device=device)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

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

        segments = result.get("segments", [])
        confidence = self._calculate_confidence(segments)

        return RecognitionResult(text=text, confidence=confidence)

    def _bytes_to_float_array(self, audio_data: bytes, sample_rate: int) -> np.ndarray[Any, Any] | None:
        if len(audio_data) == 0:
            return None

        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if sample_rate != 16000:
            audio_float32 = self._resample(audio_float32, sample_rate, 16000)

        return audio_float32

    def _resample(
        self, audio: np.ndarray[Any, Any], orig_sr: int, target_sr: int
    ) -> np.ndarray[Any, Any]:
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled: np.ndarray[Any, Any] = np.interp(
            indices, np.arange(len(audio)), audio
        ).astype(np.float32)
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
