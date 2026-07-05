"""Схлопывание дублей в распознанном тексте.

Применяется последним в пайплайне (после replacements и форматирования): именно
там появляются дубли на стыке — например, слово-команда «точка» превращается в
«.», а правило замены даёт «.env», итог «..env». Модуль убирает такие дубли
набором правил, который легко расширять — добавь `CleanupRule` в `_build_rules`.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

# Знаки, дубль которых на стыке нужно схлопывать в один.
_COLLAPSIBLE_PUNCT = r".,;:!?"

# Легитимные повторы, которые НЕЛЬЗЯ трогать (многоточие, операторы).
_PROTECTED_SEQUENCES = ("...", "--", "===", "==", "->", "=>", "::", "//")

# Токены, внутри которых нельзя ничего схлопывать (URL, email, код).
_PROTECTED_TOKEN = re.compile(
    r"""
    \S*                     # опциональный префикс
    (?:
        [a-z]+://\S+        # схема://...
      | \S+@\S+\.\S+        # email
      | \S*(?:->|=>|::)\S*  # код-операторы внутри токена
    )
    \S*
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class CleanupRule:
    """Одно правило очистки. Расширяемо: добавь экземпляр в TextCleaner._build_rules."""

    name: str
    apply: Callable[[str], str]


def _collapse_duplicate_punctuation(text: str) -> str:
    """`..env`→`.env`, `. .env`→`.env`, `,,`→`,`; `...` и операторы защищены плейсхолдером."""

    def repl(match: re.Match[str]) -> str:
        return match.group(1)

    # Один и тот же знак подряд (с возможным пробелом между): `..`, `. .`, `,,`.
    return re.sub(rf"([{re.escape(_COLLAPSIBLE_PUNCT)}])(?:\s*\1)+", repl, text)


def _collapse_spoken_symbol_before_symbol(text: str) -> str:
    """«точка .env»→`.env`, «запятая , два»→`, два` — осиротевшее слово-команда."""
    words = {
        "точка": ".",
        "запятая": ",",
        "двоеточие": ":",
        "точка с запятой": ";",
    }
    for word, symbol in words.items():
        # «слово <пробел> символ» → символ (слово-команда продублировало знак)
        pattern = re.compile(
            rf"\b{word}\s+(?=\{symbol})",
            re.IGNORECASE,
        )
        text = pattern.sub("", text)
    return text


def _collapse_repeated_words(text: str) -> str:
    """«коммит коммит»→«коммит» (регистронезависимо, по границам слов)."""
    return re.sub(
        r"\b(\w+)(\s+\1\b)+",
        lambda m: m.group(1),
        text,
        flags=re.IGNORECASE,
    )


def _normalize_whitespace(text: str) -> str:
    """Двойные пробелы → один; убрать пробел перед пунктуацией; обрезать края."""
    text = re.sub(r"\s+", " ", text)
    # Пробел перед знаком убираем, только если это пунктуация конца клаузы
    # (за знаком пробел/конец/закрывающая скобка), а не имя вроде «.env» (за знаком буква).
    text = re.sub(
        rf"\s+([{re.escape(_COLLAPSIBLE_PUNCT)}])(?=\s|$|[)\]}}»\"'])",
        r"\1",
        text,
    )
    return text.strip()


class TextCleaner:
    def __init__(self) -> None:
        self.rules = self._build_rules()

    def _build_rules(self) -> list[CleanupRule]:
        return [
            CleanupRule("spoken_symbol_before_symbol", _collapse_spoken_symbol_before_symbol),
            CleanupRule("duplicate_punctuation", _collapse_duplicate_punctuation),
            CleanupRule("repeated_words", _collapse_repeated_words),
            CleanupRule("whitespace", _normalize_whitespace),
        ]

    def clean(self, text: str) -> str:
        if not text:
            return text
        protected, restored = self._protect_tokens(text)
        for rule in self.rules:
            protected = rule.apply(protected)
        return self._restore_tokens(protected, restored)

    def _protect_tokens(self, text: str) -> tuple[str, list[str]]:
        """Заменяет URL/email/код-токены плейсхолдерами, чтобы правила их не трогали."""
        restored: list[str] = []

        def stash(match: re.Match[str]) -> str:
            restored.append(match.group(0))
            return f"\x00{len(restored) - 1}\x00"

        protected = _PROTECTED_TOKEN.sub(stash, text)
        for seq in _PROTECTED_SEQUENCES:
            protected = self._stash_sequence(protected, seq, restored)
        return protected, restored

    def _stash_sequence(self, text: str, seq: str, restored: list[str]) -> str:
        def stash(match: re.Match[str]) -> str:
            restored.append(match.group(0))
            return f"\x00{len(restored) - 1}\x00"

        return re.sub(re.escape(seq), stash, text)

    def _restore_tokens(self, text: str, restored: list[str]) -> str:
        for i, original in enumerate(restored):
            text = text.replace(f"\x00{i}\x00", original)
        return text
