"""Baseline corrector that returns text unchanged — the zero-damage reference."""

from __future__ import annotations


class NoOpCorrector:
    @property
    def name(self) -> str:
        return "noop"

    def setup(self) -> None:
        return None

    def correct(self, text: str) -> str:
        return text
