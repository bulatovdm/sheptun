"""Orchestrates a benchmark run: samples → correctors → metrics → report."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypeAlias

from .metrics import DEFAULT_TERMS, damage, exact_match_rate, latin_tokens, term_tokens
from .types import (
    BatchCorrector,
    BenchmarkReport,
    CorrectionResult,
    Corrector,
    CorrectorReport,
    Sample,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Reports progress after each corrected line: (corrector_name, done, total).
ProgressCallback: TypeAlias = "Callable[[str, int, int], None]"

_MAX_EXAMPLES = 8
_PROGRESS_CHUNK = 16


def _correct_all(
    corrector: Corrector,
    samples: Sequence[Sample],
    on_progress: ProgressCallback | None,
) -> list[str]:
    """Correct every sample, using batch mode when the corrector supports it.

    Batch correctors report progress in chunks (per-line ticks aren't observable
    inside a batched forward pass); per-line correctors tick every line.
    """
    texts = [s.text for s in samples]
    total = len(texts)

    if isinstance(corrector, BatchCorrector):
        outputs: list[str] = []
        for start in range(0, total, _PROGRESS_CHUNK):
            chunk = texts[start : start + _PROGRESS_CHUNK]
            outputs.extend(corrector.correct_batch(chunk))
            if on_progress is not None:
                on_progress(corrector.name, min(start + len(chunk), total), total)
        return outputs

    result: list[str] = []
    for index, text in enumerate(texts, start=1):
        result.append(corrector.correct(text))
        if on_progress is not None:
            on_progress(corrector.name, index, total)
    return result


def _run_corrector(
    corrector: Corrector,
    samples: Sequence[Sample],
    terms: frozenset[str],
    on_progress: ProgressCallback | None = None,
) -> CorrectorReport:
    corrector.setup()

    total = len(samples)
    started = time.perf_counter()
    outputs = _correct_all(corrector, samples, on_progress)
    ms_per_line = (time.perf_counter() - started) / total * 1000 if samples else 0.0

    results: list[CorrectionResult] = []
    for sample, output in zip(samples, outputs, strict=True):
        lost_latin, lost_terms = damage(sample.text, output, terms)
        results.append(
            CorrectionResult(
                sample=sample, output=output, lost_latin=lost_latin, lost_terms=lost_terms
            )
        )

    ref_pairs = [(r.output, r.sample.reference) for r in results if r.sample.reference is not None]
    examples = tuple(r for r in results if r.damaged)[:_MAX_EXAMPLES]

    return CorrectorReport(
        name=corrector.name,
        total=len(samples),
        changed=sum(1 for r in results if r.changed),
        damaged=sum(1 for r in results if r.damaged),
        lost_latin=sum(len(r.lost_latin) for r in results),
        lost_terms=sum(len(r.lost_terms) for r in results),
        ms_per_line=ms_per_line,
        exact_match=exact_match_rate((o, r) for o, r in ref_pairs),
        examples=examples,
    )


def run(
    samples: Sequence[Sample],
    correctors: Sequence[Corrector],
    terms: frozenset[str] = DEFAULT_TERMS,
    on_progress: ProgressCallback | None = None,
) -> BenchmarkReport:
    return BenchmarkReport(
        sample_count=len(samples),
        with_latin=sum(1 for s in samples if latin_tokens(s.text)),
        with_terms=sum(1 for s in samples if term_tokens(s.text, terms)),
        with_reference=sum(1 for s in samples if s.reference is not None),
        correctors=tuple(_run_corrector(c, samples, terms, on_progress) for c in correctors),
    )
