from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.samples import from_log, from_replacements

if TYPE_CHECKING:
    from pathlib import Path

SAMPLE_LOG = """\
2026-07-01 10:31:15,491 [INFO] Recognized: 'подними докер контейнер быстро'
2026-07-01 10:31:20,000 [INFO] Window changed: A -> B
2026-07-01 10:31:22,238 [INFO] Recognized: 'да'
2026-07-01 10:31:26,000 [INFO] Recognized: 'сделай коммит и запушь в мейн'
"""


class TestFromLog:
    def test_keeps_only_command_sized_recognized_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "sheptun.log"
        log.write_text(SAMPLE_LOG, encoding="utf-8")
        samples = from_log(log, count=10, min_words=3)
        texts = {s.text for s in samples}
        assert "подними докер контейнер быстро" in texts
        assert "сделай коммит и запушь в мейн" in texts
        assert "да" not in texts  # too short (below min_words)

    def test_count_limits_and_no_reference(self, tmp_path: Path) -> None:
        log = tmp_path / "sheptun.log"
        log.write_text(SAMPLE_LOG, encoding="utf-8")
        samples = from_log(log, count=1)
        assert len(samples) == 1
        assert samples[0].reference is None

    def test_seed_is_deterministic(self, tmp_path: Path) -> None:
        log = tmp_path / "sheptun.log"
        log.write_text(SAMPLE_LOG, encoding="utf-8")
        assert [s.text for s in from_log(log, 5, seed=1)] == [
            s.text for s in from_log(log, 5, seed=1)
        ]

    def test_count_zero_takes_all(self, tmp_path: Path) -> None:
        log = tmp_path / "sheptun.log"
        log.write_text(SAMPLE_LOG, encoding="utf-8")
        # 2 command-sized lines survive the word filter; count=0 → all of them
        assert len(from_log(log, count=0)) == 2

    def test_dedup_removes_case_insensitive_duplicates(self, tmp_path: Path) -> None:
        log = tmp_path / "sheptun.log"
        dup = (
            "2026-01-01 10:00:00,000 [INFO] Recognized: 'сделай коммит и пуш'\n"
            "2026-01-01 10:00:01,000 [INFO] Recognized: 'Сделай Коммит И Пуш'\n"
            "2026-01-01 10:00:02,000 [INFO] Recognized: 'подними докер быстро сейчас'\n"
        )
        log.write_text(dup, encoding="utf-8")
        assert len(from_log(log, count=0, dedup=True)) == 2
        assert len(from_log(log, count=0, dedup=False)) == 3


class TestFromReplacements:
    def test_builds_pairs_and_skips_multiword_keys(self, tmp_path: Path) -> None:
        repl = tmp_path / "replacements.yaml"
        repl.write_text(
            '"комит": "коммит"\n'
            '"докер": "docker"\n'
            '"сделай комит": "сделай коммит"\n',  # multi-word key → skipped
            encoding="utf-8",
        )
        samples = from_replacements(repl)
        keys = {s.text for s in samples}
        assert keys == {"комит", "докер"}
        by_key = {s.text: s.reference for s in samples}
        assert by_key["комит"] == "коммит"
        assert by_key["докер"] == "docker"

    def test_empty_when_not_a_mapping(self, tmp_path: Path) -> None:
        repl = tmp_path / "replacements.yaml"
        repl.write_text("- just\n- a\n- list\n", encoding="utf-8")
        assert from_replacements(repl) == []
