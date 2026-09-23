import json
import sqlite3
from pathlib import Path

import pytest

from sheptun.lm_corpus import (
    build_corpus,
    normalize_sentences,
    normalize_word,
    read_recognized_phrases,
    read_replacement_values,
    read_testset_sentences,
    read_verified_transcripts,
)


class TestNormalize:
    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("Commit.", "commit"),
            ("«Ёлка»,", "елка"),
            ("README.md", "readme.md"),
            (".env", "env"),
            ("—", ""),
        ],
    )
    def test_word(self, word: str, expected: str) -> None:
        assert normalize_word(word) == expected

    def test_splits_sentences_and_drops_punctuation_only(self) -> None:
        text = "Сделай git push. Потом — проверь! …"
        assert normalize_sentences(text) == ["сделай git push", "потом проверь"]


class TestReaders:
    def test_recognized_phrases_skip_replacement_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "sheptun.log"
        log.write_text(
            "2026-09-01 10:00:00,000 [INFO] Recognized: 'Сделай гид пуш.'\n"
            "2026-09-01 10:00:00,001 [INFO] Recognized: 'гид' -> 'git'\n"
            "2026-09-01 10:00:00,002 [DEBUG] Clipboard send complete\n",
            encoding="utf-8",
        )
        assert list(read_recognized_phrases(log)) == ["Сделай гид пуш."]

    def test_verified_transcripts_skip_hallucinations(self, tmp_path: Path) -> None:
        db_path = tmp_path / "verification.db"
        with sqlite3.connect(db_path) as db:
            db.execute(
                "CREATE TABLE verifications (verified_text TEXT, corrected_text TEXT, "
                "is_hallucination INTEGER)"
            )
            db.executemany(
                "INSERT INTO verifications VALUES (?, ?, ?)",
                [("Верно.", None, 0), (None, "Исправлено.", 0), ("Субтитры.", None, 1)],
            )
        assert sorted(read_verified_transcripts(db_path)) == ["Верно.", "Исправлено."]

    def test_missing_sources_are_empty(self, tmp_path: Path) -> None:
        assert list(read_verified_transcripts(tmp_path / "absent.db")) == []
        assert list(read_replacement_values(tmp_path / "absent.yaml")) == []
        assert read_testset_sentences(tmp_path / "absent.jsonl") == set()

    def test_replacement_values(self, tmp_path: Path) -> None:
        path = tmp_path / "replacements.yaml"
        path.write_text('"гид": "git"\n"пуш": "push"\n', encoding="utf-8")
        assert list(read_replacement_values(path)) == ["git", "push"]

    def test_testset_keeps_only_long_sentences(self, tmp_path: Path) -> None:
        path = tmp_path / "references.jsonl"
        rows = [{"id": "01", "text": "Сохранить."}, {"id": "02", "text": "Сделай git push."}]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert read_testset_sentences(path) == {"сделай git push"}


class TestBuildCorpus:
    def test_applies_replacements_and_adds_extra_texts(self) -> None:
        corpus = build_corpus(
            ["Сделай гид пуш."],
            lambda text: text.replace("гид пуш", "git push"),
            ["README.md"],
            excluded=set(),
        )
        assert corpus == ["сделай git push", "readme.md"]

    def test_drops_paraphrases_of_testset_references(self) -> None:
        corpus = build_corpus(
            ["Так деплой на стейдж, потом на prod.", "Деплой завтра."],
            lambda text: text,
            [],
            excluded={"так деплой на staging потом на prod"},
        )
        assert corpus == ["деплой завтра"]

    def test_drops_sentences_containing_testset_references(self) -> None:
        corpus = build_corpus(
            ["Ну сделай git push сейчас.", "Открой файл."],
            lambda text: text,
            [],
            excluded={"сделай git push"},
        )
        assert corpus == ["открой файл"]
