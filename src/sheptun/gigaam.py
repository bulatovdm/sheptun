import logging
import threading
import time
from typing import Any, Protocol

import numpy as np

from sheptun.recognition import _bytes_to_float_array, _filter_hallucination, _WarmupMixin
from sheptun.settings import settings
from sheptun.types import RecognitionResult

logger = logging.getLogger("sheptun")

_SAMPLE_RATE = 16000
_WARMUP_SILENCE_FRAMES = 1600


class _Runtime(Protocol):
    def transcribe(self, audio_array: np.ndarray[Any, Any]) -> str: ...


class _MlxRuntime:
    """GigaAM on Apple GPU via mlx. Roughly 3x faster than the ONNX CPU runtime."""

    def __init__(self, model_name: str) -> None:
        try:
            from gigaam_mlx import compute_mel, load_model  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "gigaam-mlx не установлен. Установите: pip install -e '.[gigaam-mlx]'"
            ) from e

        import mlx.core as mx  # type: ignore[import-untyped]

        self._mx = mx
        self._compute_mel = compute_mel
        self._model, self._tokenizer = load_model("rnnt" if "rnnt" in model_name else "ctc")

    def transcribe(self, audio_array: np.ndarray[Any, Any]) -> str:
        mel = self._compute_mel(audio_array)
        encoded, seq_len = self._model.encode(self._mx.array(mel[np.newaxis]))
        self._mx.eval(encoded)
        token_ids = self._model.decode(encoded, seq_len)
        return str(self._tokenizer.decode(token_ids)).strip()


class _OnnxRuntime:
    """GigaAM on CPU via onnx-asr. Portable fallback when mlx is unavailable."""

    def __init__(self, model_name: str, quantization: str | None) -> None:
        try:
            import onnx_asr  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "onnx-asr не установлен. Установите: pip install -e '.[gigaam]'"
            ) from e

        self._model: Any = onnx_asr.load_model(
            model_name, quantization=quantization, providers=["CPUExecutionProvider"]
        )

    def transcribe(self, audio_array: np.ndarray[Any, Any]) -> str:
        return str(self._model.recognize(audio_array, sample_rate=_SAMPLE_RATE)).strip()


class GigaAMRecognizer(_WarmupMixin):
    """GigaAM v3 e2e: Russian with punctuation, casing and numerals out of the box."""

    def __init__(
        self,
        model_name: str | None = None,
        runtime: str | None = None,
        quantization: str | None = None,
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        self._model_name = model_name or settings.gigaam_model
        self._runtime_name = runtime or settings.gigaam_runtime
        self._runtime = self._create_runtime(quantization)
        self._hallucinations = {h.lower() for h in (hallucinations or settings.hallucinations)}
        self._transcribe_lock = threading.Lock()
        self._init_warmup(warmup_interval)

    def _create_runtime(self, quantization: str | None) -> _Runtime:
        if self._runtime_name == "mlx":
            return _MlxRuntime(self._model_name)
        return _OnnxRuntime(self._model_name, quantization or settings.gigaam_quantization or None)

    @property
    def model_name(self) -> str:
        return f"{self._model_name} ({self._runtime_name})"

    def _transcribe(self, audio_array: np.ndarray[Any, Any]) -> str:
        started = time.monotonic()
        text = self._runtime.transcribe(audio_array)
        duration = len(audio_array) / _SAMPLE_RATE
        logger.debug(
            f"GigaAM input: {duration:.2f}s audio, inference {time.monotonic() - started:.2f}s"
        )
        return text

    def _do_warmup(self) -> None:
        warmup_audio = np.zeros(_WARMUP_SILENCE_FRAMES, dtype=np.float32)
        with self._transcribe_lock:
            self._transcribe(warmup_audio)

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        # GigaAM теряет ~2 п.п. CER на обрезанных краях, Whisper к этому безразличен
        audio_array = _bytes_to_float_array(audio_data, sample_rate, trim=False)
        if audio_array is None:
            return None

        with self._transcribe_lock:
            text = self._transcribe(audio_array)

        if not text:
            logger.debug("GigaAM returned empty text")
            return None

        filtered = _filter_hallucination(text, self._hallucinations)
        if filtered is None:
            return None

        logger.debug(f"GigaAM recognized: {filtered}")
        return RecognitionResult(text=filtered, confidence=1.0)
