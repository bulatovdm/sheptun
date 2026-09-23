from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("pyctcdecode")
pytest.importorskip("kenlm")

from sheptun.gigaam import load_language_model
from sheptun.ngram_lm import (
    NormalizedLanguageModel,
    train_language_model,
    unigrams_path,
)


class _FakeKenLM:
    order = 4

    def __init__(self, vocabulary: set[str]) -> None:
        self._vocabulary = vocabulary
        self.scored: list[str] = []

    def __contains__(self, word: str) -> bool:
        return word in self._vocabulary

    def BaseScore(self, _prev_state: Any, word: str, _end_state: Any) -> float:
        self.scored.append(word)
        return -1.0

    def BeginSentenceWrite(self, _state: Any) -> None:
        pass


@pytest.fixture
def kenlm_model() -> _FakeKenLM:
    return _FakeKenLM({"git", "commit"})


@pytest.fixture
def language_model(kenlm_model: _FakeKenLM) -> NormalizedLanguageModel:
    return NormalizedLanguageModel(kenlm_model, ["git", "commit"], alpha=0.5, beta=1.0)


class TestNormalizedLanguageModel:
    def test_scores_normalized_word(
        self, language_model: NormalizedLanguageModel, kenlm_model: _FakeKenLM
    ) -> None:
        language_model.score(language_model.get_start_state(), "Commit.")
        assert kenlm_model.scored == ["commit"]

    def test_punctuation_only_word_is_free(
        self, language_model: NormalizedLanguageModel, kenlm_model: _FakeKenLM
    ) -> None:
        state = language_model.get_start_state()
        assert language_model.score(state, "—") == (0.0, state)
        assert kenlm_model.scored == []

    def test_partial_token_matched_case_insensitively(
        self, language_model: NormalizedLanguageModel
    ) -> None:
        assert language_model.score_partial_token("Com") == 0.0
        assert language_model.score_partial_token("Жук") < 0.0


class TestTrainLanguageModel:
    def test_writes_unigrams_and_removes_arpa(self, tmp_path: Path) -> None:
        output = tmp_path / "lm.bin"

        def fake_run(cmd: list[str], **_: Any) -> None:
            if Path(cmd[0]).name == "build_binary":
                Path(cmd[2]).write_bytes(b"binary")

        with patch("sheptun.ngram_lm.subprocess.run", side_effect=fake_run):
            train_language_model(
                ["сделай git push", "git commit"],
                output,
                order=4,
                lmplz=Path("lmplz"),
                build_binary=Path("build_binary"),
            )

        assert output.exists()
        assert not output.with_suffix(".arpa").exists()
        assert unigrams_path(output).read_text(encoding="utf-8").split() == [
            "commit",
            "git",
            "push",
            "сделай",
        ]


class TestLoadLanguageModel:
    def test_disabled_without_path(self) -> None:
        with patch("sheptun.gigaam.settings", gigaam_lm_path=None):
            assert load_language_model() is None

    def test_missing_file_is_skipped(self, tmp_path: Path) -> None:
        with patch("sheptun.gigaam.settings", gigaam_lm_path=str(tmp_path / "absent.bin")):
            assert load_language_model() is None
