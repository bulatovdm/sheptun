import logging
import re
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
_LATIN_TERM = re.compile(r"[A-Za-z][\w.+#/-]*")


def load_hotwords(limit: int) -> list[str]:
    """Английские термины из replacements.yaml — подсказки декодеру, чтобы не ушли в транслит."""
    import yaml

    from sheptun.config import get_replacements_path

    path = get_replacements_path()
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    if not isinstance(rules, dict):
        return []

    terms = (v for v in rules.values() if isinstance(v, str) and _LATIN_TERM.fullmatch(v))
    return list(dict.fromkeys(terms))[:limit]


class _HotwordDecoder:
    """CTC beam search с бустом терминов: 'Gid Comiт' → 'Git commit' ценой ~90ms на фразу."""

    def __init__(self, tokenizer: Any, hotwords: list[str]) -> None:
        try:
            from pyctcdecode import build_ctcdecoder  # type: ignore[import-untyped, attr-defined]
        except ImportError as e:
            raise ImportError(
                "pyctcdecode не установлен. Установите: pip install -e '.[gigaam-mlx]'"
            ) from e

        labels = [tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())] + [""]
        self._decoder = build_ctcdecoder(labels)
        self._hotwords = hotwords

    def decode(self, log_probs: np.ndarray[Any, Any]) -> str:
        text = self._decoder.decode(
            log_probs,
            beam_width=settings.gigaam_beam_width,
            hotwords=self._hotwords or None,
            hotword_weight=settings.gigaam_hotword_weight,
        )
        return str(text).strip()


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
        self._model_type = "rnnt" if "rnnt" in model_name else "ctc"
        self._model, self._tokenizer = load_model(self._model_type)
        self._decoder = self._create_decoder()

    def _create_decoder(self) -> _HotwordDecoder | None:
        if not settings.gigaam_hotwords:
            return None
        if self._model_type != "ctc":
            logger.info("GigaAM hotwords работают только с CTC-моделью, биасинг выключен")
            return None

        hotwords = load_hotwords(settings.gigaam_hotwords_limit)
        if not hotwords:
            return None

        logger.info(f"GigaAM hotwords: {len(hotwords)} терминов из replacements.yaml")
        return _HotwordDecoder(self._tokenizer, hotwords)

    def transcribe(self, audio_array: np.ndarray[Any, Any]) -> str:
        mel = self._compute_mel(audio_array)
        encoded, seq_len = self._model.encode(self._mx.array(mel[np.newaxis]))
        self._mx.eval(encoded)

        if self._decoder is None:
            token_ids = self._model.decode(encoded, seq_len)
            return str(self._tokenizer.decode(token_ids)).strip()

        log_probs = self._model.head(encoded)
        self._mx.eval(log_probs)
        return self._decoder.decode(np.array(log_probs)[0, :seq_len, :])


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
