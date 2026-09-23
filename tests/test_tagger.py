import random
from pathlib import Path
from typing import ClassVar

import pytest

from sheptun.tagger_data import (
    Example,
    ExampleBuilder,
    TermVocabulary,
    build_examples,
    label_rule_matches,
    transliterate,
)
from sheptun.term_tagger import (
    DELETE,
    FIRST_TERM_LABEL,
    KEEP,
    TermCorrector,
    apply_labels,
    create_term_corrector,
    split_token,
)

RULES = {"гид": "git", "пуш": "push", "тейп скрипт": "TypeScript", "проерь": "проверь"}


@pytest.fixture
def vocab() -> TermVocabulary:
    return TermVocabulary(RULES)


def _label(vocab: TermVocabulary, term: str) -> int:
    return vocab.label[term]


class TestSplitToken:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("«Commit,", ("«", "Commit", ",")),
            ("README.md.", ("", "README.md", ".")),
            ("—", ("—", "", "")),
        ],
    )
    def test_splits_edge_punctuation(self, token: str, expected: tuple[str, str, str]) -> None:
        assert split_token(token) == expected


class TestApplyLabels:
    TARGETS: ClassVar[list[str]] = ["TypeScript", "git"]

    def test_replaces_keeping_punctuation(self) -> None:
        tokens = ["Сделай", "«гид,"]
        text = apply_labels(tokens, [KEEP, FIRST_TERM_LABEL + 1], [1.0, 0.9], self.TARGETS, 0.5)
        assert text == "Сделай «git,"

    def test_merges_deleted_words_into_replacement(self) -> None:
        tokens = ["на", "тейп", "скрипт."]
        labels = [KEEP, FIRST_TERM_LABEL, DELETE]
        assert apply_labels(tokens, labels, [1.0, 0.9, 0.9], self.TARGETS, 0.5) == "на TypeScript."

    def test_unsure_edit_keeps_original(self) -> None:
        tokens = ["гид"]
        assert apply_labels(tokens, [FIRST_TERM_LABEL + 1], [0.4], self.TARGETS, 0.5) == "гид"

    def test_corrector_uses_predictor(self) -> None:
        corrector = TermCorrector(
            predict=lambda _tokens: ([KEEP, FIRST_TERM_LABEL + 1], [1.0, 0.8]),
            targets=self.TARGETS,
            threshold=0.5,
        )
        assert corrector.correct("Сделай гид") == "Сделай git"


class TestCreateCorrector:
    def test_disabled_without_path(self) -> None:
        assert create_term_corrector(None, 0.5) is None

    def test_missing_model_is_skipped(self, tmp_path: Path) -> None:
        assert create_term_corrector(str(tmp_path / "absent"), 0.5) is None


class TestData:
    def test_transliterates_like_asr(self) -> None:
        assert transliterate("push") == "пуш"
        assert transliterate("commit") == "коммит"

    def test_finds_multiword_terms(self, vocab: TermVocabulary) -> None:
        assert vocab.find(["на", "TypeScript,", "и", "git"]) == {
            1: (1, "TypeScript"),
            3: (1, "git"),
        }

    def test_labels_rule_matches_with_merge(self, vocab: TermVocabulary) -> None:
        line = "пиши на тейп скрипт"
        spans = [(8, 19, "TypeScript")]
        labels = label_rule_matches(line.split(), spans, line, vocab)
        assert labels == [KEEP, KEEP, _label(vocab, "TypeScript"), DELETE]

    def test_raw_line_uses_rule_output_as_label(self, vocab: TermVocabulary) -> None:
        builder = ExampleBuilder(vocab, lambda _line: [(8, 11, "git")], set(), random.Random(0))
        example = builder.from_raw("Сделай гид сейчас")
        assert example == Example(["Сделай", "гид", "сейчас"], [KEEP, _label(vocab, "git"), KEEP])

    def test_corruption_keeps_labels_aligned(self, vocab: TermVocabulary) -> None:
        builder = ExampleBuilder(vocab, lambda _line: [], set(), random.Random(1))
        examples = [e for _ in range(20) for e in builder.from_clean("Сделай git push сейчас.")]
        edited = [e for e in examples if any(label != KEEP for label in e.labels)]
        assert edited
        for example in examples:
            assert len(example.tokens) == len(example.labels)
            assert example.tokens[-1].endswith("сейчас.")

    def test_russian_terms_get_only_real_distortions(self, vocab: TermVocabulary) -> None:
        builder = ExampleBuilder(vocab, lambda _line: [], set(), random.Random(2))
        examples = [e for _ in range(30) for e in builder.from_clean("Ну проверь это")]
        corrupted = {e.tokens[1] for e in examples if e.labels[1] != KEEP}
        assert corrupted <= {"проерь", "Проерь"}

    def test_testset_phrases_are_excluded(self, vocab: TermVocabulary) -> None:
        builder = ExampleBuilder(
            vocab, lambda _line: [], {"сделай git push сейчас"}, random.Random(0)
        )
        assert builder.from_clean("Сделай git push сейчас.") == []

    def test_build_examples_combines_sources(self, vocab: TermVocabulary) -> None:
        builder = ExampleBuilder(
            vocab,
            lambda line: [(0, 3, "git")] if line.startswith("гид") else [],
            set(),
            random.Random(0),
        )
        examples = build_examples(["гид готов."], lambda t: t.replace("гид", "git"), [], builder)
        assert Example(["гид", "готов."], [_label(vocab, "git"), KEEP]) in examples


class TestModel:
    def test_trains_saves_and_predicts(self, tmp_path: Path, vocab: TermVocabulary) -> None:
        pytest.importorskip("mlx")
        from sheptun.tagger_model import EpochReport, load_predictor, train_tagger

        git = _label(vocab, "git")
        examples = [Example(["Сделай", "гид", "сейчас"], [KEEP, git, KEEP])] * 8
        reports: list[EpochReport] = []
        train_tagger(
            examples, examples[:2], vocab.targets, tmp_path, epochs=2, on_epoch=reports.append
        )

        assert [r.epoch for r in reports] == [1, 2]
        predict = load_predictor(tmp_path, len(vocab.targets))
        labels, confidence = predict(["Сделай", "гид"])
        assert len(labels) == len(confidence) == 2
        assert all(0.0 <= c <= 1.0 for c in confidence)
