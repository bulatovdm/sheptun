from __future__ import annotations

import pytest

from benchmarks import correctors
from benchmarks.correctors.noop import NoOpCorrector


class TestRegistry:
    def test_available_lists_known_correctors(self) -> None:
        assert set(correctors.available()) == {"noop", "jamspell", "sage"}

    def test_create_returns_instance(self) -> None:
        assert isinstance(correctors.create("noop"), NoOpCorrector)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Неизвестный корректор"):
            correctors.create("nope")


class TestNoOp:
    def test_returns_text_unchanged(self) -> None:
        c = NoOpCorrector()
        c.setup()
        assert c.correct("сделай коммит") == "сделай коммит"


class TestHeavyCorrectorsFailFastWithoutSetup:
    def test_jamspell_correct_without_setup_raises(self) -> None:
        from benchmarks.correctors.jamspell import JamSpellCorrector

        with pytest.raises(RuntimeError, match="setup"):
            JamSpellCorrector().correct("текст")

    def test_sage_correct_without_setup_raises(self) -> None:
        from benchmarks.correctors.sage import SageCorrector

        with pytest.raises(RuntimeError, match="setup"):
            SageCorrector().correct("текст")
