# pyright: reportPrivateUsage=false
from unittest.mock import MagicMock, patch

from sheptun.spelling import (
    NoOpCorrector,
    SpellCorrectorType,
    create_corrector,
)


class TestNoOpCorrector:
    def test_returns_same_text(self) -> None:
        corrector = NoOpCorrector()
        assert corrector.correct("привет") == "привет"

    def test_returns_empty_string(self) -> None:
        corrector = NoOpCorrector()
        assert corrector.correct("") == ""


class TestSpellCorrectorType:
    def test_none_value(self) -> None:
        assert SpellCorrectorType.NONE.value == "none"

    def test_t5_russian_value(self) -> None:
        assert SpellCorrectorType.T5_RUSSIAN.value == "t5-russian"


class TestCreateCorrector:
    def test_creates_noop_for_none(self) -> None:
        corrector = create_corrector(SpellCorrectorType.NONE)
        assert isinstance(corrector, NoOpCorrector)

    @patch("sheptun.spelling.T5RussianCorrector")
    def test_creates_t5_russian(self, mock_t5: MagicMock) -> None:
        create_corrector(SpellCorrectorType.T5_RUSSIAN)
        mock_t5.assert_called_once()


class TestT5RussianCorrectorMocked:
    @patch("sheptun.spelling.T5RussianCorrector.__init__", return_value=None)
    def test_correct_returns_original_on_empty(self, _mock_init: MagicMock) -> None:
        from sheptun.spelling import T5RussianCorrector

        corrector = T5RussianCorrector()
        corrector._model = MagicMock()
        corrector._tokenizer = MagicMock()

        result = corrector.correct("")
        assert result == ""
        corrector._model.generate.assert_not_called()
