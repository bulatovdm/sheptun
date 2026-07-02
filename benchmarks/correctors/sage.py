"""SAGE corrector — SberDevices generative spell/punct/case model for Russian.

Uses the distilled 95M FRED-T5 variant (~0.38 GB) via HuggingFace Transformers.

Tunables (env) for finding the optimal production config:
- ``SHEPTUN_BENCH_SAGE_MODEL``  — model id (default: distilled 95M)
- ``SHEPTUN_BENCH_SAGE_DEVICE`` — auto | mps | cuda | cpu (default: auto)
- ``SHEPTUN_BENCH_SAGE_BEAMS``  — beam count; 1 = greedy, ~3x faster (default: 1)
- ``SHEPTUN_BENCH_SAGE_BATCH``  — batch size for correct_batch (default: 16)
"""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_MODEL = "ai-forever/sage-fredt5-distilled-95m"
_LEN_FACTOR = 1.5
_LEN_PAD = 10
_MAX_LEN = 300
_INPUT_MAX_LEN = 256


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SageCorrector:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device = "cpu"
        self._beams = int(os.environ.get("SHEPTUN_BENCH_SAGE_BEAMS", "1"))
        self._batch_size = int(os.environ.get("SHEPTUN_BENCH_SAGE_BATCH", "16"))

    @property
    def name(self) -> str:
        return "sage"

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise RuntimeError(
                "Пакеты 'transformers'/'torch' не установлены: pip install -e '.[bench]'"
            ) from exc

        model_name = os.environ.get("SHEPTUN_BENCH_SAGE_MODEL", _DEFAULT_MODEL)
        self._device = _resolve_device(os.environ.get("SHEPTUN_BENCH_SAGE_DEVICE", "auto"))
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        self._model = model.to(self._device)
        del torch

    def _generate(self, texts: list[str]) -> list[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("SageCorrector.setup() не вызван")
        import torch

        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=_INPUT_MAX_LEN,
            padding=True,
        ).to(self._device)
        max_len = min(int(encoded.input_ids.shape[1] * _LEN_FACTOR) + _LEN_PAD, _MAX_LEN)
        with torch.no_grad():
            generated = self._model.generate(**encoded, max_length=max_len, num_beams=self._beams)
        return [str(t) for t in self._tokenizer.batch_decode(generated, skip_special_tokens=True)]

    def correct(self, text: str) -> str:
        return self._generate([text])[0]

    def correct_batch(self, texts: list[str]) -> list[str]:
        results: list[str] = []
        for start in range(0, len(texts), self._batch_size):
            results.extend(self._generate(texts[start : start + self._batch_size]))
        return results
