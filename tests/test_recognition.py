# pyright: reportPrivateUsage=false
import pytest

from sheptun.recognition import (
    _FOREIGN_SCRIPT_PATTERN,
    _REPETITIVE_PATTERN,
    _SOUND_DESCRIPTION_PATTERN,
    WhisperRecognizer,
)


class TestHallucinationFiltering:
    @pytest.fixture
    def recognizer(self) -> WhisperRecognizer:
        return WhisperRecognizer(
            model_name="tiny",
            hallucinations=(
                "Продолжение следует...",
                "Спасибо за просмотр!",
                "Test Hallucination",
            ),
        )

    def test_is_hallucination_exact_match(self, recognizer: WhisperRecognizer) -> None:
        assert recognizer._is_hallucination("Продолжение следует...") is True

    def test_is_hallucination_case_insensitive(self, recognizer: WhisperRecognizer) -> None:
        assert recognizer._is_hallucination("ПРОДОЛЖЕНИЕ СЛЕДУЕТ...") is True
        assert recognizer._is_hallucination("test hallucination") is True
        assert recognizer._is_hallucination("TEST HALLUCINATION") is True

    def test_is_hallucination_not_matching(self, recognizer: WhisperRecognizer) -> None:
        assert recognizer._is_hallucination("Привет мир") is False
        assert recognizer._is_hallucination("клод") is False

    def test_is_hallucination_partial_match_not_filtered(
        self, recognizer: WhisperRecognizer
    ) -> None:
        assert recognizer._is_hallucination("Продолжение") is False
        assert recognizer._is_hallucination("Спасибо") is False

    def test_custom_hallucinations_override_defaults(self) -> None:
        recognizer = WhisperRecognizer(
            model_name="tiny",
            hallucinations=("Custom phrase",),
        )
        assert recognizer._is_hallucination("Custom phrase") is True
        assert recognizer._is_hallucination("Продолжение следует...") is False


class TestWhisperRecognizerInit:
    def test_default_hallucinations_from_settings(self) -> None:
        recognizer = WhisperRecognizer(model_name="tiny")
        assert recognizer._is_hallucination("Продолжение следует...") is True

    def test_hallucinations_stored_as_set(self) -> None:
        recognizer = WhisperRecognizer(
            model_name="tiny",
            hallucinations=("test", "TEST", "Test"),
        )
        assert len(recognizer._hallucinations) == 1


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
