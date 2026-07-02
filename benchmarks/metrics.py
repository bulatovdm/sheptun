"""Metrics for judging a corrector on mixed Russian/English technical speech.

The central concern for Sheptun is DAMAGE: a corrector trained on plain Russian
may "fix" a valid English term into a Russian word (докер→уокер, git→гит), which
is worse than leaving a typo. We measure damage as English/technical tokens present
in the input but gone from the output.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_LATIN = re.compile(r"[A-Za-z]{2,}")

# Cyrillic transliterations of English tech terms that live in replacements.yaml as
# "mush → English". A corrector must not mangle these into unrelated Russian words.
DEFAULT_TERMS: frozenset[str] = frozenset(
    {
        "докер",
        "гит",
        "коммит",
        "пуш",
        "пул",
        "деплой",
        "бэкенд",
        "фронтенд",
        "пайтон",
        "джанго",
        "ларавел",
        "нжинкс",
        "апи",
        "джейсон",
        "реакт",
        "вью",
        "мейн",
        "стейджинг",
        "продакшн",
        "линтер",
        "рефактор",
        "конфиг",
        "билд",
        "мердж",
        "ребейз",
        "воркер",
        "миграции",
        "роут",
        "мидлвар",
        "эндпоинт",
    }
)


def latin_tokens(text: str) -> frozenset[str]:
    return frozenset(m.group(0).lower() for m in _LATIN.finditer(text))


def term_tokens(text: str, terms: frozenset[str]) -> frozenset[str]:
    low = text.lower()
    return frozenset(t for t in terms if re.search(rf"\b{re.escape(t)}\b", low))


def damage(orig: str, fixed: str, terms: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    """English tokens and tech-terms present in ``orig`` but missing from ``fixed``."""
    lost_latin = latin_tokens(orig) - latin_tokens(fixed)
    lost_terms = term_tokens(orig, terms) - term_tokens(fixed, terms)
    return lost_latin, lost_terms


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def exact_match_rate(pairs: Iterable[tuple[str, str]]) -> float | None:
    """Share of outputs that equal their reference (case/space-insensitive).

    Returns None when no pair is given (no reference available for the set).
    """
    matched = 0
    total = 0
    for output, reference in pairs:
        total += 1
        if _normalize(output) == _normalize(reference):
            matched += 1
    return matched / total if total else None
