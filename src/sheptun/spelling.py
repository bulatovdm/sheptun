import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

logger = logging.getLogger("sheptun.spelling")


class SpellCorrectorType(Enum):
    NONE = "none"
    T5_RUSSIAN = "t5-russian"  # 200M, UrukHan/t5-russian-spell


_T5_RUSSIAN_MODEL = "UrukHan/t5-russian-spell"


class SpellCorrector(ABC):
    @abstractmethod
    def correct(self, text: str) -> str:
        pass


class NoOpCorrector(SpellCorrector):
    def correct(self, text: str) -> str:
        return text


class T5RussianCorrector(SpellCorrector):
    def __init__(self) -> None:
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForSeq2SeqLM,
            T5TokenizerFast,
        )

        logger.info(f"Loading T5 model: {_T5_RUSSIAN_MODEL}")
        self._tokenizer: Any = T5TokenizerFast.from_pretrained(_T5_RUSSIAN_MODEL)
        self._model: Any = AutoModelForSeq2SeqLM.from_pretrained(_T5_RUSSIAN_MODEL)
        self._prefix = "Spell correct: "
        logger.info("T5 model loaded")

    def correct(self, text: str) -> str:
        if not text.strip():
            return text

        try:
            encoded = self._tokenizer(
                self._prefix + text,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            predictions = self._model.generate(**encoded)
            results = self._tokenizer.batch_decode(predictions, skip_special_tokens=True)
            if results and len(results) > 0:
                corrected = results[0]
                if corrected != text:
                    logger.debug(f"Spell corrected: '{text}' -> '{corrected}'")
                return str(corrected)
        except Exception as e:
            logger.warning(f"Spell correction failed: {e}")

        return text


_corrector_instance: SpellCorrector | None = None


def get_corrector() -> SpellCorrector:
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = create_corrector()
    return _corrector_instance


def create_corrector(corrector_type: SpellCorrectorType | None = None) -> SpellCorrector:
    from sheptun.settings import settings

    if corrector_type is None:
        corrector_type = SpellCorrectorType(settings.spell_correction)

    if corrector_type == SpellCorrectorType.NONE:
        return NoOpCorrector()

    if corrector_type == SpellCorrectorType.T5_RUSSIAN:
        return T5RussianCorrector()

    return NoOpCorrector()


def correct_text(text: str) -> str:
    return get_corrector().correct(text)


def download_model() -> None:
    """Download spell correction model to cache."""
    from sheptun.settings import settings

    corrector_type = SpellCorrectorType(settings.spell_correction)

    if corrector_type == SpellCorrectorType.NONE:
        return

    if corrector_type == SpellCorrectorType.T5_RUSSIAN:
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForSeq2SeqLM,
            T5TokenizerFast,
        )

        logger.info(f"Downloading T5 model: {_T5_RUSSIAN_MODEL}")
        T5TokenizerFast.from_pretrained(_T5_RUSSIAN_MODEL)
        AutoModelForSeq2SeqLM.from_pretrained(_T5_RUSSIAN_MODEL)
        logger.info("T5 model downloaded")
