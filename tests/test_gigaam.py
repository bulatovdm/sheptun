# pyright: reportPrivateUsage=false
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sheptun.benchmark import _compute_epi
from sheptun.gigaam import load_hotwords, split_at_pauses


def _write_replacements(path: Path, rules: dict[str, str]) -> None:
    lines = [f'"{key}": "{value}"' for key, value in rules.items()]
    path.write_text("\n".join(lines), encoding="utf-8")


class TestLoadHotwords:
    def test_keeps_only_latin_values(self, tmp_path: Path) -> None:
        path = tmp_path / "replacements.yaml"
        _write_replacements(
            path,
            {
                "комит": "commit",
                "продакшен": "продакшн",
                "гитхаб": "GitHub",
                "точка енв": ".env",
            },
        )

        with patch("sheptun.config.get_replacements_path", return_value=path):
            assert load_hotwords(10) == ["commit", "GitHub"]

    def test_deduplicates_and_respects_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "replacements.yaml"
        _write_replacements(
            path,
            {"комит": "commit", "комитт": "commit", "пуш": "push", "чекаут": "checkout"},
        )

        with patch("sheptun.config.get_replacements_path", return_value=path):
            assert load_hotwords(2) == ["commit", "push"]

    def test_missing_file_gives_no_hotwords(self, tmp_path: Path) -> None:
        with patch("sheptun.config.get_replacements_path", return_value=tmp_path / "absent.yaml"):
            assert load_hotwords(10) == []


class TestComputeEpi:
    @pytest.mark.parametrize(
        ("hypothesis", "expected"),
        [
            ("Сделай git commit", 1.0),  # оба термина точно
            ("Сделай git comit", 0.75),  # git точно + опечатка в одну букву
            ("Сделай git com", 0.625),  # git точно + латинский огрызок
            ("Сделай гит комит", 0.0),  # всё ушло в транслит
        ],
    )
    def test_scores_latin_preservation(self, hypothesis: str, expected: float) -> None:
        assert _compute_epi(hypothesis, "Сделай git commit") == pytest.approx(expected)

    def test_no_latin_terms_is_not_scored(self) -> None:
        assert _compute_epi("Открой терминал", "Открой терминал") is None


class TestSplitAtPauses:
    BLANK = 2

    def _frames(self, labels: list[int]) -> np.ndarray:
        frames = np.full((len(labels), 3), -10.0)
        frames[np.arange(len(labels)), labels] = 0.0
        return frames

    def test_cuts_in_the_middle_of_long_pauses(self) -> None:
        labels = [0, 1] + [self.BLANK] * 10 + [1, 0]
        parts = split_at_pauses(self._frames(labels), self.BLANK)
        assert [len(p) for p in parts] == [7, 7]

    def test_short_pauses_stay_whole(self) -> None:
        labels = [0] + [self.BLANK] * 3 + [1]
        assert len(split_at_pauses(self._frames(labels), self.BLANK)) == 1
