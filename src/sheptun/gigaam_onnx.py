import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from sheptun.recognition import _bytes_to_float_array, _filter_hallucination, _WarmupMixin
from sheptun.settings import settings
from sheptun.types import RecognitionResult

logger = logging.getLogger("sheptun")

_REPO_ID = "i2z1/gigaam-multilingual-ctc-onnx-int8"
_MODEL_FILE = "model.int8.onnx"
_TOKENS_FILE = "tokens.txt"
_SAMPLE_RATE = 16000
_N_MELS = 64
_N_FFT = 400
_HOP_LENGTH = 160
_WARMUP_SILENCE_FRAMES = 1600
_LOG_CLAMP_MIN = 1e-9
_LOG_CLAMP_MAX = 1e9


class GigaAMRecognizer(_WarmupMixin):
    """GigaAM Multilingual CTC (220M) via onnxruntime: ru/en/kk/ky/uz, no punctuation."""

    def __init__(
        self,
        repo_id: str = _REPO_ID,
        hallucinations: tuple[str, ...] | None = None,
        warmup_interval: float | None = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "onnxruntime не установлен. Установите: pip install onnxruntime"
            ) from e

        model_path, tokens_path = self._download(repo_id)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._tokens = self._load_tokens(tokens_path)
        self._blank_id = len(self._tokens) - 1
        self._repo_id = repo_id
        self._hallucinations = {h.lower() for h in (hallucinations or settings.hallucinations)}
        self._featurizer = self._build_featurizer()
        self._transcribe_lock = threading.Lock()
        self._init_warmup(warmup_interval)

    @property
    def model_name(self) -> str:
        return self._repo_id

    @staticmethod
    def _download(repo_id: str) -> tuple[Path, Path]:
        from huggingface_hub import hf_hub_download

        return (
            Path(hf_hub_download(repo_id, _MODEL_FILE)),
            Path(hf_hub_download(repo_id, _TOKENS_FILE)),
        )

    @staticmethod
    def _load_tokens(tokens_path: Path) -> list[str]:
        tokens: list[str] = []
        for line in tokens_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            separator = line.rfind(" ")
            tokens.append(line[:separator])
        return tokens

    @staticmethod
    def _build_featurizer() -> Any:
        import torchaudio  # type: ignore[import-untyped]

        return torchaudio.transforms.MelSpectrogram(
            sample_rate=_SAMPLE_RATE,
            n_mels=_N_MELS,
            n_fft=_N_FFT,
            win_length=_N_FFT,
            hop_length=_HOP_LENGTH,
            center=True,
        )

    def _extract_features(self, audio_array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        import torch

        with torch.no_grad():
            mel = self._featurizer(torch.from_numpy(audio_array))
            features = torch.log(mel.clamp(_LOG_CLAMP_MIN, _LOG_CLAMP_MAX))
        return features.unsqueeze(0).numpy().astype(np.float32)

    def _decode(self, log_probs: np.ndarray[Any, Any], length: int) -> str:
        best_ids = log_probs[0, :length].argmax(axis=-1)
        pieces: list[str] = []
        previous = -1
        for token_id in best_ids:
            if token_id != previous and token_id != self._blank_id:
                pieces.append(self._tokens[token_id])
            previous = int(token_id)
        return "".join(pieces).strip()

    def _transcribe(self, audio_array: np.ndarray[Any, Any]) -> str:
        features = self._extract_features(audio_array)
        lengths = np.array([len(audio_array) // _HOP_LENGTH + 1], dtype=np.int64)
        outputs = self._session.run(None, {"features": features, "feature_lengths": lengths})
        log_probs = np.asarray(outputs[0])
        encoded_lengths = np.asarray(outputs[1])
        return self._decode(log_probs, int(encoded_lengths[0]))

    def _do_warmup(self) -> None:
        warmup_audio = np.zeros(_WARMUP_SILENCE_FRAMES, dtype=np.float32)
        with self._transcribe_lock:
            self._transcribe(warmup_audio)

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        audio_array = _bytes_to_float_array(audio_data, sample_rate)
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
