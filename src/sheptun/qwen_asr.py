import logging
import threading
from typing import Any

import numpy as np

from sheptun.recognition import _bytes_to_float_array, _check_hallucination, _WarmupMixin
from sheptun.settings import settings
from sheptun.types import RecognitionResult

logger = logging.getLogger("sheptun")

_DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
_WARMUP_SILENCE_FRAMES = 1600


class QwenASRRecognizer(_WarmupMixin):
    """Speech recognizer using Qwen3-ASR via mlx-qwen3-asr on Apple Silicon."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        try:
            from mlx_qwen3_asr import Session  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "mlx-qwen3-asr не установлен. Установите: pip install mlx-qwen3-asr"
            ) from e

        self._session: Any = Session(model=model_id)
        self._model_id = model_id
        self._hallucinations = {h.lower() for h in (hallucinations or settings.hallucinations)}
        self._transcribe_lock = threading.Lock()
        self._init_warmup(warmup_interval)

    @property
    def model_name(self) -> str:
        return self._model_id

    def _do_warmup(self) -> None:
        warmup_audio = np.zeros(_WARMUP_SILENCE_FRAMES, dtype=np.float32)
        with self._transcribe_lock:
            self._session.transcribe((warmup_audio, 16000))

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        audio_array = _bytes_to_float_array(audio_data, sample_rate)
        if audio_array is None:
            return None

        duration = len(audio_array) / 16000
        logger.debug(f"Qwen3-ASR input: {duration:.2f}s audio ({len(audio_data)} bytes raw)")

        with self._transcribe_lock:
            result = self._session.transcribe(
                (audio_array, 16000),
                language="Russian",
            )

        text = result.text.strip() if hasattr(result, "text") else str(result).strip()
        if not text:
            logger.debug("Qwen3-ASR returned empty text")
            return None

        if _check_hallucination(text, self._hallucinations):
            logger.debug(f"Hallucination filtered: '{text}'")
            return None

        return RecognitionResult(text=text, confidence=1.0)
