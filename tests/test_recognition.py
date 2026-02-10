# pyright: reportPrivateUsage=false
from typing import Any

import pytest

from sheptun.recognition import (
    _FOREIGN_SCRIPT_PATTERN,
    _REPETITIVE_PATTERN,
    _SOUND_DESCRIPTION_PATTERN,
    MLX_MODELS,
    _calculate_confidence,
    _check_hallucination,
    resolve_mlx_model,
)


class TestHallucinationFiltering:
    @pytest.fixture
    def hallucinations(self) -> set[str]:
        return {
            "продолжение следует...",
            "спасибо за просмотр!",
            "test hallucination",
        }

    def test_is_hallucination_exact_match(self, hallucinations: set[str]) -> None:
        assert _check_hallucination("Продолжение следует...", hallucinations) is True

    def test_is_hallucination_case_insensitive(self, hallucinations: set[str]) -> None:
        assert _check_hallucination("ПРОДОЛЖЕНИЕ СЛЕДУЕТ...", hallucinations) is True
        assert _check_hallucination("test hallucination", hallucinations) is True
        assert _check_hallucination("TEST HALLUCINATION", hallucinations) is True

    def test_is_hallucination_not_matching(self, hallucinations: set[str]) -> None:
        assert _check_hallucination("Привет мир", hallucinations) is False
        assert _check_hallucination("клод", hallucinations) is False

    def test_is_hallucination_partial_match_not_filtered(self, hallucinations: set[str]) -> None:
        assert _check_hallucination("Продолжение", hallucinations) is False
        assert _check_hallucination("Спасибо", hallucinations) is False

    def test_custom_hallucinations_override_defaults(self) -> None:
        custom = {"custom phrase"}
        assert _check_hallucination("Custom phrase", custom) is True
        assert _check_hallucination("Продолжение следует...", custom) is False


class TestWhisperRecognizerInit:
    def test_default_hallucinations(self) -> None:
        from sheptun.settings import settings

        hallucinations = {h.lower() for h in settings.hallucinations}
        assert _check_hallucination("Продолжение следует...", hallucinations) is True

    def test_hallucinations_set_dedup(self) -> None:
        hallucinations = {h.lower() for h in ("test", "TEST", "Test")}
        assert len(hallucinations) == 1


class TestSoundDescriptionPattern:
    @pytest.mark.parametrize(
        "text",
        [
            "СПОКОЙНАЯ МУЗЫКА",
            "СМЕХ",
            "ДИНАМИЧНАЯ МУЗЫКА",
            "СТУК В ДВЕРЬ",
            "СМЕХ СМЕХ СМЕХ",
            "АПЛОДИСМЕНТЫ",
        ],
    )
    def test_matches_cyrillic_caps(self, text: str) -> None:
        assert _SOUND_DESCRIPTION_PATTERN.match(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Привет",
            "СМЕХ Привет",
            "Привет СМЕХ",
            "давай",
            "Hello",
            "HELLO WORLD",
        ],
    )
    def test_does_not_match_normal_text(self, text: str) -> None:
        assert not _SOUND_DESCRIPTION_PATTERN.match(text)


class TestRepetitivePattern:
    @pytest.mark.parametrize(
        "text",
        [
            "а, а, а, а, а, а, а",
            "о, о, о, о, о, о",
            "э м О, а, а, а, а, а, а, а, а",
            "ха, ха, ха, ха, ха, ха",
        ],
    )
    def test_matches_repetitive_syllables(self, text: str) -> None:
        assert _REPETITIVE_PATTERN.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Привет, как дела",
            "да, да, нет",
            "один, два, три, четыре",
            "а, б, в, г, д",
        ],
    )
    def test_does_not_match_normal_text(self, text: str) -> None:
        assert not _REPETITIVE_PATTERN.search(text)


class TestForeignScriptPattern:
    @pytest.mark.parametrize(
        "text",
        [
            "του Xiaomi",
            "integrating σум",
            "это 中文 текст",
            "арабский العربية",
            "японский の日本語",
        ],
    )
    def test_matches_foreign_scripts(self, text: str) -> None:
        assert _FOREIGN_SCRIPT_PATTERN.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Привет, как дела",
            "Hello world",
            "Тест 123",
            "Mixed Текст",
        ],
    )
    def test_does_not_match_cyrillic_latin(self, text: str) -> None:
        assert not _FOREIGN_SCRIPT_PATTERN.search(text)


class TestCalculateConfidence:
    def test_empty_segments(self) -> None:
        assert _calculate_confidence([]) == 0.0

    def test_single_segment(self) -> None:
        segments: list[dict[str, Any]] = [
            {"avg_logprob": -0.5, "no_speech_prob": 0.1, "tokens": [1, 2, 3]},
        ]
        confidence = _calculate_confidence(segments)
        assert 0.0 < confidence < 1.0

    def test_skips_no_speech_segments(self) -> None:
        segments: list[dict[str, Any]] = [
            {"avg_logprob": -0.5, "no_speech_prob": 0.9, "tokens": [1, 2]},
        ]
        assert _calculate_confidence(segments) == 0.0

    def test_multiple_segments(self) -> None:
        segments: list[dict[str, Any]] = [
            {"avg_logprob": -0.3, "no_speech_prob": 0.1, "tokens": [1, 2]},
            {"avg_logprob": -0.5, "no_speech_prob": 0.2, "tokens": [3, 4, 5]},
        ]
        confidence = _calculate_confidence(segments)
        assert 0.0 < confidence < 1.0


class TestMLXModels:
    def test_all_standard_models_present(self) -> None:
        for name in ("tiny", "base", "small", "medium", "large", "turbo"):
            assert name in MLX_MODELS

    def test_models_point_to_mlx_community(self) -> None:
        for repo in MLX_MODELS.values():
            assert repo.startswith("mlx-community/whisper-")

    def test_resolve_mlx_model_known(self) -> None:
        assert resolve_mlx_model("turbo") == "mlx-community/whisper-large-v3-turbo"
        assert resolve_mlx_model("base") == "mlx-community/whisper-base"

    def test_resolve_mlx_model_passthrough(self) -> None:
        custom = "mlx-community/whisper-custom"
        assert resolve_mlx_model(custom) == custom

    def test_resolve_mlx_model_unknown_passthrough(self) -> None:
        assert resolve_mlx_model("unknown-model") == "unknown-model"
