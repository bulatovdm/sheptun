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

    def test_sage_distilled_value(self) -> None:
        assert SpellCorrectorType.SAGE_DISTILLED.value == "sage-distilled"

    def test_sage_large_value(self) -> None:
        assert SpellCorrectorType.SAGE_LARGE.value == "sage-large"

    def test_t5_russian_value(self) -> None:
        assert SpellCorrectorType.T5_RUSSIAN.value == "t5-russian"


class TestCreateCorrector:
    def test_creates_noop_for_none(self) -> None:
        corrector = create_corrector(SpellCorrectorType.NONE)
        assert isinstance(corrector, NoOpCorrector)

    @patch("sheptun.spelling.SageCorrector")
    def test_creates_sage_distilled(self, mock_sage: MagicMock) -> None:
        create_corrector(SpellCorrectorType.SAGE_DISTILLED)
        mock_sage.assert_called_once_with("sage-distilled")

    @patch("sheptun.spelling.SageCorrector")
    def test_creates_sage_large(self, mock_sage: MagicMock) -> None:
        create_corrector(SpellCorrectorType.SAGE_LARGE)
        mock_sage.assert_called_once_with("sage-large")

    @patch("sheptun.spelling.T5RussianCorrector")
    def test_creates_t5_russian(self, mock_t5: MagicMock) -> None:
        create_corrector(SpellCorrectorType.T5_RUSSIAN)
        mock_t5.assert_called_once()


class TestSageCorrectorMocked:
    @patch("sheptun.spelling.SageCorrector.__init__", return_value=None)
    def test_correct_calls_model(self, _mock_init: MagicMock) -> None:
        from sheptun.spelling import SageCorrector

        corrector = SageCorrector("sage-large")
        corrector._model = MagicMock()
        corrector._model.correct.return_value = ["исправленный текст"]

        result = corrector.correct("текст с ошибкой")

        corrector._model.correct.assert_called_once_with("текст с ошибкой")
        assert result == "исправленный текст"

    @patch("sheptun.spelling.SageCorrector.__init__", return_value=None)
    def test_correct_returns_original_on_empty(self, _mock_init: MagicMock) -> None:
        from sheptun.spelling import SageCorrector

        corrector = SageCorrector("sage-large")
        corrector._model = MagicMock()

        result = corrector.correct("")
        assert result == ""
        corrector._model.correct.assert_not_called()

    @patch("sheptun.spelling.SageCorrector.__init__", return_value=None)
    def test_correct_returns_original_on_error(self, _mock_init: MagicMock) -> None:
        from sheptun.spelling import SageCorrector

        corrector = SageCorrector("sage-large")
        corrector._model = MagicMock()
        corrector._model.correct.side_effect = Exception("Model error")

        result = corrector.correct("текст")
        assert result == "текст"


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
