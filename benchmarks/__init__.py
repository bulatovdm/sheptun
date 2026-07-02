"""Text-correction benchmarks for Sheptun.

Compares correctors (SAGE, JamSpell, …) on real ASR logs, focusing on damage to
mixed Russian/English technical speech. Run: ``python -m benchmarks run``.
"""

from __future__ import annotations

from .runner import run
from .types import (
    BenchmarkReport,
    CorrectionResult,
    Corrector,
    CorrectorReport,
    Sample,
)

__all__ = [
    "BenchmarkReport",
    "CorrectionResult",
    "Corrector",
    "CorrectorReport",
    "Sample",
    "run",
]
