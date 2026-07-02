"""Build benchmark samples from real Sheptun data.

Two sources:
- ``from_log``: real Recognized lines (via the analyzer's LogParser), no reference —
  measures behaviour on live speech, esp. damage to English terms.
- ``from_replacements``: synthetic (distorted → correct) pairs straight from
  replacements.yaml — a free partial ground truth for accuracy metrics.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import yaml

from sheptun.log_analyzer import LogParser

from .types import Sample

if TYPE_CHECKING:
    from pathlib import Path

_MIN_WORDS = 3
_MAX_WORDS = 25


def from_log(
    log_path: Path,
    count: int = 0,
    seed: int = 42,
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
    dedup: bool = True,
) -> list[Sample]:
    """Recognized lines filtered to command-sized phrases.

    ``count=0`` takes the WHOLE log (no limit). ``dedup`` keeps only distinct phrases
    (case-insensitive) — the same command dictated 400 times is one correction to
    judge, so dedup makes a full run far cheaper and statistically representative.
    """
    entries = LogParser().parse(log_path)
    pool = [e.text for e in entries if min_words <= len(e.text.split()) <= max_words]
    if dedup:
        pool = _dedup_preserve_order(pool)
    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = pool if count <= 0 else pool[:count]
    return [Sample(text=t) for t in selected]


def _dedup_preserve_order(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for t in texts:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def from_replacements(
    replacements_path: Path,
    count: int | None = None,
    seed: int = 42,
) -> list[Sample]:
    """Synthetic (distorted → correct) pairs from replacements.yaml as ground truth.

    Each rule ``"old": "new"`` becomes a sample where the distorted key is the input
    and the value is the reference — a corrector should turn ``old`` into ``new``.
    Multi-word or punctuation-bearing keys are skipped (not single-token typos).
    """
    loaded = yaml.safe_load(replacements_path.read_text(encoding="utf-8"))
    rules = dict(loaded) if isinstance(loaded, dict) else {}
    samples = [
        Sample(text=str(old), reference=str(new))
        for old, new in rules.items()
        if str(old).strip() and " " not in str(old)
    ]
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:count] if count is not None else samples
