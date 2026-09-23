"""Text corpus for the GigaAM n-gram language model.

Sources: every recognized phrase from the app log with the *current* replacements.yaml
applied (rules found later also clean old lines), transcripts verified by Claude, and the
replacement values themselves. Words are normalized the same way the decoder normalizes
them at scoring time, so the LM never sees casing or edge punctuation.
"""

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import yaml

_EDGE_PUNCTUATION = ".,!?;:«»\"'()[]…—–-"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_RECOGNIZED_LINE = re.compile(r"Recognized: '(.*)'$")
# When post-processing changed the text the engine logs "'raw' -> 'processed'"
_PROCESSED_SUFFIX = re.compile(r"' -> '.*$")
# Shorter reference sentences ("Сохранить.") are too common to count as a leak
_MIN_LEAK_WORDS = 3
# A corpus sentence holding this share of a reference's words is a paraphrase of it
_LEAK_WORD_SHARE = 0.7


def normalize_word(word: str) -> str:
    return word.strip(_EDGE_PUNCTUATION).lower().replace("ё", "е")


def normalize_sentences(text: str) -> list[str]:
    sentences = []
    for part in _SENTENCE_BOUNDARY.split(text):
        words = [w for w in (normalize_word(raw) for raw in part.split()) if w]
        if words:
            sentences.append(" ".join(words))
    return sentences


def read_recognized_phrases(log_path: Path) -> Iterator[str]:
    with log_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _RECOGNIZED_LINE.search(line.rstrip())
            if match:
                yield _PROCESSED_SUFFIX.sub("", match.group(1))


def read_verified_transcripts(db_path: Path) -> Iterator[str]:
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT COALESCE(verified_text, corrected_text) FROM verifications "
            "WHERE COALESCE(is_hallucination, 0) = 0 "
            "AND COALESCE(verified_text, corrected_text) IS NOT NULL"
        )
        yield from (text for (text,) in rows)


def read_replacement_values(replacements_path: Path) -> Iterator[str]:
    if not replacements_path.exists():
        return
    with replacements_path.open(encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    if isinstance(rules, dict):
        yield from (v for v in rules.values() if isinstance(v, str) and v.strip())


def read_testset_sentences(references_path: Path) -> set[str]:
    if not references_path.exists():
        return set()
    sentences: set[str] = set()
    for line in references_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sentences.update(normalize_sentences(json.loads(line)["text"]))
    return {s for s in sentences if len(s.split()) >= _MIN_LEAK_WORDS}


class LeakFilter:
    """Spots corpus sentences that repeat a benchmark reference, even with ASR variations."""

    def __init__(self, references: set[str]) -> None:
        self._references = [set(ref.split()) for ref in references]
        self._by_word: dict[str, list[int]] = {}
        for index, words in enumerate(self._references):
            for word in words:
                self._by_word.setdefault(word, []).append(index)

    def is_leak(self, sentence: str) -> bool:
        words = set(sentence.split())
        shared: dict[int, int] = {}
        for word in words:
            for index in self._by_word.get(word, ()):
                shared[index] = shared.get(index, 0) + 1
        return any(
            count >= _LEAK_WORD_SHARE * len(self._references[index])
            for index, count in shared.items()
        )


def build_corpus(
    phrases: Iterable[str],
    apply_replacements: Callable[[str], str],
    extra_texts: Iterable[str],
    excluded: set[str],
) -> list[str]:
    """Normalized sentences; anything repeating a benchmark reference is dropped."""
    texts = [apply_replacements(p) for p in phrases]
    texts.extend(extra_texts)
    leak_filter = LeakFilter(excluded)
    corpus = (s for text in texts for s in normalize_sentences(text))
    return [s for s in corpus if not leak_filter.is_leak(s)]
