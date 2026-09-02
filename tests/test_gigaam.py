# pyright: reportPrivateUsage=false
from pathlib import Path
from unittest.mock import patch

import pytest

from sheptun.benchmark import _compute_epi
from sheptun.gigaam import load_hotwords


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
