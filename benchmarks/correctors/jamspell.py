"""JamSpell corrector — fast n-gram statistical spell checker.

Needs a pre-trained model file (Russian ``ru.tar.gz`` → ``.bin``); its path comes
from ``SHEPTUN_BENCH_JAMSPELL_MODEL``. JamSpell wheels build only against SWIG 3,
so install is manual — see benchmarks/README.md.
"""

from __future__ import annotations

import os
from typing import Any


class JamSpellCorrector:
    def __init__(self) -> None:
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return "jamspell"

    def setup(self) -> None:
        try:
            import jamspell
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise RuntimeError(
                "Пакет 'jamspell' не установлен. См. benchmarks/README.md (нужен SWIG 3)."
            ) from exc

        model_path = os.environ.get("SHEPTUN_BENCH_JAMSPELL_MODEL")
        if not model_path:
            raise RuntimeError(
                "Задай SHEPTUN_BENCH_JAMSPELL_MODEL — путь к .bin модели JamSpell (ru)."
            )
        corrector = jamspell.TSpellCorrector()
        if not corrector.LoadLangModel(model_path):
            raise RuntimeError(f"Не удалось загрузить модель JamSpell: {model_path}")
        self._model = corrector

    def correct(self, text: str) -> str:
        if self._model is None:
            raise RuntimeError("JamSpellCorrector.setup() не вызван")
        return str(self._model.FixFragment(text))
