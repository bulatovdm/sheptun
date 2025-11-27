# pyright: reportPrivateUsage=false
import pytest

from sheptun.recognition import WhisperRecognizer


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
