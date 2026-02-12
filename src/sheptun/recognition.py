import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import whisper

from sheptun.settings import settings
from sheptun.types import RecognitionResult

logger = logging.getLogger("sheptun")

_SOUND_DESCRIPTION_PATTERN = re.compile(r"^[А-ЯЁ\s]+$")
_REPETITIVE_PATTERN = re.compile(r"(.{1,3},\s*)\1{4,}")
_FOREIGN_SCRIPT_PATTERN = re.compile(r"[\u0370-\u03FF\u4E00-\u9FFF\u0600-\u06FF\u3040-\u30FF]")
_WARMUP_AUDIO = np.zeros(1600, dtype=np.float32)


def is_local_model(model_name: str) -> bool:
    return Path(model_name).is_dir()


def _check_hallucination(text: str, hallucinations: set[str]) -> bool:
    text_lower = text.lower()
    if any(h in text_lower for h in hallucinations):
        return True
    text_stripped = text.strip()
    if _SOUND_DESCRIPTION_PATTERN.match(text_stripped):
        return True
    if _REPETITIVE_PATTERN.search(text_stripped):
        return True
    return bool(_FOREIGN_SCRIPT_PATTERN.search(text_stripped))


def _bytes_to_float_array(audio_data: bytes, sample_rate: int) -> np.ndarray[Any, Any] | None:
    if len(audio_data) == 0:
        return None

    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    if sample_rate != 16000:
        audio_float32 = _resample(audio_float32, sample_rate, 16000)

    audio_float32 = _trim_silence(audio_float32, sample_rate)
    return audio_float32


def _trim_silence(
    audio: np.ndarray[Any, Any],
    sample_rate: int,
    threshold: float = 0.01,
    window_ms: int = 20,
) -> np.ndarray[Any, Any]:
    window_size = int(sample_rate * window_ms / 1000)
    if len(audio) < window_size:
        return audio

    start = _find_speech_boundary(audio, window_size, threshold, sample_rate, from_start=True)
    end = _find_speech_boundary(audio, window_size, threshold, sample_rate, from_start=False)

    trimmed = audio[start:end]
    trimmed_duration = (len(audio) - len(trimmed)) / sample_rate
    if trimmed_duration > 0.1:
        logger.debug(f"Trimmed {trimmed_duration:.2f}s silence")

    return trimmed


def _find_speech_boundary(
    audio: np.ndarray[Any, Any],
    window_size: int,
    threshold: float,
    sample_rate: int,
    *,
    from_start: bool,
) -> int:
    if from_start:
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i : i + window_size]
            energy = np.sqrt(np.mean(window**2))
            if energy > threshold:
                return max(0, i - int(sample_rate * 0.05))
        return 0

    for i in range(len(audio) - window_size, 0, -window_size):
        window = audio[i : i + window_size]
        energy = np.sqrt(np.mean(window**2))
        if energy > threshold:
            return min(len(audio), i + window_size + int(sample_rate * 0.05))
    return len(audio)


def _resample(audio: np.ndarray[Any, Any], orig_sr: int, target_sr: int) -> np.ndarray[Any, Any]:
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled: np.ndarray[Any, Any] = np.interp(indices, np.arange(len(audio)), audio).astype(
        np.float32
    )
    return resampled


def _apply_spell_correction(text: str) -> str:
    from sheptun.spelling import correct_text

    return correct_text(text)


def _calculate_confidence(segments: list[dict[str, Any]]) -> float:
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


class _WarmupMixin:
    _warmup_interval: float
    _warmup_timer: threading.Timer | None
    _warmup_lock: threading.Lock

    def _init_warmup(self, warmup_interval: float | None) -> None:
        self._warmup_interval = (
            warmup_interval if warmup_interval is not None else settings.warmup_interval
        )
        self._warmup_timer = None
        self._warmup_lock = threading.Lock()

    def _do_warmup(self) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        try:
            self._do_warmup()
            logger.debug("Warmup completed")
        except Exception as e:
            logger.debug(f"Warmup error: {e}")

    def start_warmup(self) -> None:
        if self._warmup_interval <= 0:
            return
        self._schedule_warmup()
        logger.debug(f"Warmup started with interval {self._warmup_interval}s")

    def stop_warmup(self) -> None:
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


class WhisperRecognizer(_WarmupMixin):
    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        self._model: Any = whisper.load_model(model_name, device=device)
        self._model_name = model_name
        self._hallucinations = {h.lower() for h in (hallucinations or settings.hallucinations)}
        self._init_warmup(warmup_interval)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _do_warmup(self) -> None:
        self._model.transcribe(_WARMUP_AUDIO, language="ru", fp16=False)

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        audio_array = _bytes_to_float_array(audio_data, sample_rate)
        if audio_array is None:
            return None

        result = self._model.transcribe(
            audio_array,
            language="ru",
            fp16=False,
            task="transcribe",
            condition_on_previous_text=False,
        )

        original_text = result.get("text", "").strip()
        if not original_text:
            logger.debug("Whisper returned empty text")
            return None

        if _check_hallucination(original_text, self._hallucinations):
            logger.debug(f"Hallucination filtered: '{original_text}'")
            return None

        corrected_text = _apply_spell_correction(original_text)

        segments = result.get("segments", [])
        confidence = _calculate_confidence(segments)

        return RecognitionResult(
            text=corrected_text,
            confidence=confidence,
            original_text=original_text if corrected_text != original_text else None,
        )

    def recognize_from_file(self, audio_path: Path) -> RecognitionResult | None:
        result = self._model.transcribe(
            str(audio_path),
            language="ru",
            fp16=False,
            task="transcribe",
            condition_on_previous_text=False,
        )

        original_text = result.get("text", "").strip()
        if not original_text:
            return None

        if _check_hallucination(original_text, self._hallucinations):
            return None

        corrected_text = _apply_spell_correction(original_text)

        segments = result.get("segments", [])
        confidence = _calculate_confidence(segments)

        return RecognitionResult(
            text=corrected_text,
            confidence=confidence,
            original_text=original_text if corrected_text != original_text else None,
        )


class HuggingFaceWhisperRecognizer(_WarmupMixin):
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        self._model_path = model_path
        self._hallucinations = {h.lower() for h in (hallucinations or settings.hallucinations)}
        self._init_warmup(warmup_interval)

        detect_device = device or self._detect_device()

        self._pipe = _create_asr_pipeline(model=model_path, device=detect_device)

    @property
    def model_name(self) -> str:
        return self._model_path

    def _do_warmup(self) -> None:
        self._pipe(_WARMUP_AUDIO)

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        audio_array = _bytes_to_float_array(audio_data, sample_rate)
        if audio_array is None:
            return None

        result = self._pipe(audio_array, return_timestamps=False)
        text = result.get("text", "").strip()
        if not text:
            return None

        if _check_hallucination(text, self._hallucinations):
            return None

        corrected = _apply_spell_correction(text)

        return RecognitionResult(
            text=corrected,
            confidence=1.0,
            original_text=text if corrected != text else None,
        )

    @staticmethod
    def _detect_device() -> str:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


def _create_asr_pipeline(
    model: str,
    device: str,
) -> Any:
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        generate_kwargs={"language": "russian", "task": "transcribe"},
    )


MLX_MODELS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_mlx_model(model_name: str) -> str:
    return MLX_MODELS.get(model_name, model_name)


def _get_model_expected_size(repo_id: str) -> int | None:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile

        api = HfApi()
        files = api.list_repo_tree(repo_id)
        return sum(f.size for f in files if isinstance(f, RepoFile) and f.size)
    except Exception:
        pass
    return None


def _get_cache_dir_size(repo_id: str) -> int:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    folder_name = f"models--{repo_id.replace('/', '--')}"
    model_dir = cache_dir / folder_name
    if not model_dir.exists():
        return 0
    return sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())


class MLXWhisperRecognizer(_WarmupMixin):
    def __init__(
        self,
        model_name: str = "turbo",
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        self._model_repo = resolve_mlx_model(model_name)
        self._model_name = model_name
        self._hallucinations = {h.lower() for h in (hallucinations or settings.hallucinations)}
        self._transcribe_lock = threading.Lock()
        self._init_warmup(warmup_interval)

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_model_cached(self) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache

            result = try_to_load_from_cache(self._model_repo, "config.json")
            return isinstance(result, str)
        except Exception:
            return False

    def download_model(self, on_progress: Callable[[int], None] | None = None) -> None:
        import time

        from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

        logger.info(f"Downloading MLX model: {self._model_repo}")

        if on_progress is None:
            snapshot_download(repo_id=self._model_repo)  # pyright: ignore[reportUnknownMemberType]
            logger.info(f"Model downloaded: {self._model_repo}")
            return

        expected_size = _get_model_expected_size(self._model_repo)
        error: BaseException | None = None

        def _download() -> None:
            nonlocal error
            try:
                snapshot_download(repo_id=self._model_repo)  # pyright: ignore[reportUnknownMemberType]
            except BaseException as e:
                error = e

        download_thread = threading.Thread(target=_download, daemon=True)
        download_thread.start()

        last_pct = -1
        while download_thread.is_alive():
            if expected_size and expected_size > 0:
                current_size = _get_cache_dir_size(self._model_repo)
                pct = min(99, int(current_size * 100 / expected_size))
                if pct != last_pct:
                    last_pct = pct
                    on_progress(pct)
            time.sleep(0.5)

        download_thread.join()
        if error is not None:
            raise error

        on_progress(100)
        logger.info(f"Model downloaded: {self._model_repo}")

    def _do_warmup(self) -> None:
        import mlx_whisper  # type: ignore[import-untyped]

        with self._transcribe_lock:
            mlx_whisper.transcribe(  # pyright: ignore[reportUnknownMemberType]
                _WARMUP_AUDIO,
                path_or_hf_repo=self._model_repo,
                language="ru",
                fp16=True,
            )

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        import mlx_whisper  # type: ignore[import-untyped]

        audio_array = _bytes_to_float_array(audio_data, sample_rate)
        if audio_array is None:
            return None

        duration = len(audio_array) / 16000
        logger.debug(f"MLX input: {duration:.2f}s audio ({len(audio_data)} bytes raw)")

        with self._transcribe_lock:
            result: dict[str, Any] = mlx_whisper.transcribe(  # pyright: ignore[reportUnknownMemberType]
                audio_array,
                path_or_hf_repo=self._model_repo,
                language="ru",
                fp16=True,
                condition_on_previous_text=False,
            )

        original_text = result.get("text", "").strip()
        if not original_text:
            logger.debug("MLX Whisper returned empty text")
            return None

        if _check_hallucination(original_text, self._hallucinations):
            logger.debug(f"Hallucination filtered: '{original_text}'")
            return None

        corrected_text = _apply_spell_correction(original_text)

        segments = result.get("segments", [])
        confidence = _calculate_confidence(segments)

        return RecognitionResult(
            text=corrected_text,
            confidence=confidence,
            original_text=original_text if corrected_text != original_text else None,
        )
