"""Core protocols and data structures for text-correction benchmarks.

A benchmark runs a set of correctors over a set of samples and reports, per
corrector, how much it changed, how much it damaged English/technical terms, and
(when a reference is available) accuracy against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Sample:
    """One input line for a corrector.

    ``reference`` is the known-correct text when available (e.g. derived from
    replacements.yaml), else None — metrics that need ground truth are skipped.
    """

    text: str
    reference: str | None = None


class Corrector(Protocol):
    """Anything that maps a raw ASR line to a cleaned-up one.

    ``name`` labels it in reports; ``setup`` loads models lazily so importing the
    benchmark package never pulls heavy ML deps until a corrector is actually run.
    """

    @property
    def name(self) -> str: ...
    def setup(self) -> None: ...
    def correct(self, text: str) -> str: ...


@runtime_checkable
class BatchCorrector(Protocol):
    """A corrector that can process many lines at once (GPU batching).

    The runner prefers ``correct_batch`` when present — far faster for neural
    correctors — and falls back to per-line ``correct`` otherwise.
    """

    def correct_batch(self, texts: list[str]) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """One corrector's output for one sample plus the per-sample damage it caused."""

    sample: Sample
    output: str
    lost_latin: frozenset[str]
    lost_terms: frozenset[str]

    @property
    def changed(self) -> bool:
        return self.sample.text.strip() != self.output.strip()

    @property
    def damaged(self) -> bool:
        return bool(self.lost_latin or self.lost_terms)


@dataclass(frozen=True, slots=True)
class CorrectorReport:
    """Aggregated metrics for one corrector over the whole sample set."""

    name: str
    total: int
    changed: int
    damaged: int
    lost_latin: int
    lost_terms: int
    ms_per_line: float
    # Accuracy vs reference (None when no sample carried a reference).
    exact_match: float | None = None
    examples: tuple[CorrectionResult, ...] = field(default_factory=tuple)

    @property
    def changed_pct(self) -> float:
        return 100.0 * self.changed / self.total if self.total else 0.0

    @property
    def damaged_pct(self) -> float:
        return 100.0 * self.damaged / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """The full benchmark outcome: sample stats + one report per corrector."""

    sample_count: int
    with_latin: int
    with_terms: int
    with_reference: int
    correctors: tuple[CorrectorReport, ...]
