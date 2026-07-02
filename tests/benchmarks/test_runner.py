from __future__ import annotations

from benchmarks.correctors.noop import NoOpCorrector
from benchmarks.runner import run
from benchmarks.types import Sample


class _UpperCorrector:
    """Test double: uppercases text — deterministic, no heavy deps."""

    @property
    def name(self) -> str:
        return "upper"

    def setup(self) -> None:
        return None

    def correct(self, text: str) -> str:
        return text.upper()


class _MangleCorrector:
    """Test double: replaces 'докер' with 'уокер' to simulate term damage."""

    @property
    def name(self) -> str:
        return "mangle"

    def setup(self) -> None:
        return None

    def correct(self, text: str) -> str:
        return text.replace("докер", "уокер").replace("Docker", "уокер")


class _BatchCorrector:
    """Test double exposing correct_batch — runner must prefer it over correct()."""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.line_calls = 0

    @property
    def name(self) -> str:
        return "batch"

    def setup(self) -> None:
        return None

    def correct(self, text: str) -> str:
        self.line_calls += 1
        return text

    def correct_batch(self, texts: list[str]) -> list[str]:
        self.batch_calls += 1
        return list(texts)


def _samples() -> list[Sample]:
    return [
        Sample(text="подними докер контейнер"),
        Sample(text="сделай коммит"),
        Sample(text="открой Docker Desktop"),
    ]


class TestRun:
    def test_noop_changes_nothing_and_zero_damage(self) -> None:
        report = run(_samples(), [NoOpCorrector()])
        rep = report.correctors[0]
        assert rep.changed == 0
        assert rep.damaged == 0
        assert rep.lost_latin == 0
        assert rep.lost_terms == 0

    def test_sample_stats(self) -> None:
        report = run(_samples(), [NoOpCorrector()])
        assert report.sample_count == 3
        assert report.with_latin == 1  # only "Docker Desktop" line
        assert report.with_terms == 2  # "докер" + "коммит" lines
        assert report.with_reference == 0

    def test_mangle_records_term_and_latin_damage(self) -> None:
        report = run(_samples(), [_MangleCorrector()])
        rep = report.correctors[0]
        assert rep.lost_terms == 1  # "докер" mangled in line 1
        assert rep.lost_latin == 1  # "Docker" mangled in line 3
        assert rep.damaged == 2
        assert rep.examples  # damaged examples captured

    def test_upper_counts_as_changed_but_no_damage(self) -> None:
        # uppercasing keeps latin tokens (lowered in metric) → no damage
        report = run([Sample(text="открой Docker")], [_UpperCorrector()])
        rep = report.correctors[0]
        assert rep.changed == 1
        assert rep.damaged == 0

    def test_batch_corrector_uses_batch_path(self) -> None:
        corrector = _BatchCorrector()
        report = run(_samples(), [corrector])
        assert corrector.batch_calls >= 1
        assert corrector.line_calls == 0  # per-line correct() never touched
        assert report.correctors[0].total == 3

    def test_exact_match_with_reference(self) -> None:
        samples = [
            Sample(text="докер", reference="докер"),  # noop → matches
            Sample(text="комит", reference="коммит"),  # noop → mismatch
        ]
        report = run(samples, [NoOpCorrector()])
        assert report.with_reference == 2
        assert report.correctors[0].exact_match == 0.5
