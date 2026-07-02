"""Corrector implementations and a name-based registry.

Register a new corrector by adding it to ``_FACTORIES``; the CLI resolves correctors
by name so benchmarks stay declarative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .jamspell import JamSpellCorrector
from .noop import NoOpCorrector
from .sage import SageCorrector

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..types import Corrector

_FACTORIES: dict[str, Callable[[], Corrector]] = {
    "noop": NoOpCorrector,
    "jamspell": JamSpellCorrector,
    "sage": SageCorrector,
}


def available() -> tuple[str, ...]:
    return tuple(_FACTORIES)


def create(name: str) -> Corrector:
    try:
        return _FACTORIES[name]()
    except KeyError:
        raise ValueError(
            f"Неизвестный корректор: {name!r}. Доступны: {', '.join(available())}"
        ) from None


__all__ = ["JamSpellCorrector", "NoOpCorrector", "SageCorrector", "available", "create"]
