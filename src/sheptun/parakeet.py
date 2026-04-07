import logging
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from sheptun.recognition import _bytes_to_float_array, _filter_hallucination, _WarmupMixin
from sheptun.settings import settings
from sheptun.types import RecognitionResult

logger = logging.getLogger("sheptun")

_PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2
_CHANNELS = 1
_WARMUP_SILENCE_FRAMES = 1600


class ParakeetRecognizer(_WarmupMixin):
    def __init__(
        self,
        model_id: str = _PARAKEET_MODEL_ID,
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "NeMo ASR не установлен. Установите: pip install -e '.[parakeet]'"
            ) from e

        self._model: Any = nemo_asr.models.ASRModel.from_pretrained(model_id)
        self._model_id = model_id
        self._hallucinations = {h.lower() for h in (hallucinations or settings.hallucinations)}
        self._transcribe_lock = threading.Lock()
        self._init_warmup(warmup_interval)

    @property
    def model_name(self) -> str:
        return self._model_id

    def _write_temp_wav(self, audio_array: np.ndarray[Any, Any]) -> Path:
        audio_int16 = (audio_array * 32767).astype(np.int16)
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(tmp_name, "wb") as wf:
            wf.setnchannels(_CHANNELS)
            wf.setsampwidth(_SAMPLE_WIDTH)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return Path(tmp_name)

    def _do_warmup(self) -> None:
        warmup_audio = np.zeros(_WARMUP_SILENCE_FRAMES, dtype=np.float32)
        tmp_path = self._write_temp_wav(warmup_audio)
        try:
            with self._transcribe_lock:
                self._model.transcribe([str(tmp_path)])
        finally:
            tmp_path.unlink(missing_ok=True)

    def _extract_text(self, results: Any) -> str:
        if not results:
            return ""
        item = results[0]
        if hasattr(item, "text"):
            return str(item.text).strip()
        return str(item).strip()

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        audio_array = _bytes_to_float_array(audio_data, sample_rate)
        if audio_array is None:
            return None

        tmp_path = self._write_temp_wav(audio_array)
        try:
            with self._transcribe_lock:
                results = self._model.transcribe([str(tmp_path)])
        finally:
            tmp_path.unlink(missing_ok=True)

        text = self._extract_text(results)
        if not text:
            logger.debug("Parakeet returned empty text")
            return None

        filtered = _filter_hallucination(text, self._hallucinations)
        if filtered is None:
            logger.debug(f"Hallucination filtered: '{text}'")
            return None
        if filtered != text:
            logger.debug(f"Hallucination stripped: '{text}' -> '{filtered}'")
            text = filtered

        return RecognitionResult(text=text, confidence=1.0)
