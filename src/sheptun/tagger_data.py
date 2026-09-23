"""Training examples for the term tagger: per-word KEEP / DELETE / term labels.

Three kinds of examples, all from the user's own dictation:
- real log phrases where a replacements.yaml rule fires: raw ASR in, rule output as label;
- clean phrases (log with rules applied, verified transcripts) with a term corrupted the
  way ASR does it — a real distortion from the rules, a phonetic transliteration or a
  word switching alphabet midway ('Comiт');
- clean phrases untouched, so the model learns to leave correct text alone.
"""

import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sheptun.lm_corpus import LeakFilter, normalize_sentences
from sheptun.term_tagger import DELETE, FIRST_TERM_LABEL, KEEP, MAX_WORDS, split_token

_LATIN = re.compile("[A-Za-z]")
_CORRUPT_PROBABILITY = 0.6
_REAL_DISTORTION_PROBABILITY = 0.6
_CORRUPTIONS_PER_PHRASE = 2
_NEGATIVE_PROBABILITY = 0.25
_CAPITALIZE_PROBABILITY = 0.3
_MIX_SCRIPTS_PROBABILITY = 0.3
_TYPO_PROBABILITY = 0.2
_SHORT_WORD = 3

# Longest English clusters first: 'tion' must win over 't'
_EN_TO_RU = (
    ("tion", "шн"), ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("oo", "у"),
    ("ee", "и"), ("ea", "и"), ("ou", "ау"), ("ow", "оу"), ("ai", "ей"), ("ay", "ей"), ("qu", "кв"),
    ("wh", "в"), ("x", "кс"), ("j", "дж"), ("w", "в"), ("y", "и"), ("c", "к"), ("q", "к"),
    ("a", "а"), ("b", "б"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "х"), ("i", "и"),
    ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("r", "р"), ("s", "с"),
    ("t", "т"), ("u", "у"), ("v", "в"), ("z", "з"),
)  # fmt: skip
_RU_TO_EN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "sch", "ы": "y", "э": "e", "ю": "yu", "я": "ya", "ь": "", "ъ": "",
}  # fmt: skip
_SIMILAR_LETTERS = {
    "а": "о",
    "о": "а",
    "е": "и",
    "и": "е",
    "т": "д",
    "д": "т",
    "с": "з",
    "б": "п",
    "в": "ф",
}


@dataclass(frozen=True)
class Example:
    tokens: list[str]
    labels: list[int]


def transliterate(word: str) -> str:
    """English spelled the way a Russian ASR hears it: 'commit' → 'коммит'."""
    text, out, i = word.lower(), [], 0
    while i < len(text):
        for english, russian in _EN_TO_RU:
            if text.startswith(english, i):
                out.append(russian)
                i += len(english)
                break
        else:
            out.append("" if text[i].isalpha() else text[i])
            i += 1
    result = "".join(out)
    return result[:-1] if result.endswith("е") and len(result) > _SHORT_WORD else result


class Distorter:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def mix_scripts(self, text: str) -> str:
        """ASR switches alphabet midway through a word: 'Comiт'."""
        if len(text) < _SHORT_WORD:
            return text
        cut = self._rng.randint(1, len(text) - 1)
        head, tail = text[:cut], text[cut:]
        if _LATIN.search(text):
            return head + transliterate(tail)
        return "".join(_RU_TO_EN.get(ch, ch) for ch in head.lower()) + tail

    def typo(self, word: str) -> str:
        chars = list(word)
        positions = [i for i, ch in enumerate(chars) if ch.lower() in _SIMILAR_LETTERS]
        if positions:
            i = self._rng.choice(positions)
            chars[i] = _SIMILAR_LETTERS[chars[i].lower()]
        elif len(chars) > _SHORT_WORD:
            del chars[self._rng.randrange(len(chars))]
        return "".join(chars)

    def synthetic(self, term: str) -> str:
        words = [
            transliterate(w) or w for w in term.replace("-", " ").replace(".", " точка ").split()
        ]
        text = " ".join(words)
        roll = self._rng.random()
        if roll < _MIX_SCRIPTS_PROBABILITY:
            return self.mix_scripts(text)
        if roll < _MIX_SCRIPTS_PROBABILITY + _TYPO_PROBABILITY:
            return self.typo(text)
        return text

    def recase(self, text: str) -> str:
        return text.capitalize() if self._rng.random() < _CAPITALIZE_PROBABILITY else text


class TermVocabulary:
    """Target terms (replacement values) and the real distortions behind each of them."""

    def __init__(self, rules: dict[str, str]) -> None:
        self.distortions: dict[str, list[str]] = defaultdict(list)
        for key, value in rules.items():
            if key.lower() != value.lower():
                self.distortions[value].append(key)
        self.targets = sorted(set(rules.values()))
        self.label = {term: i + FIRST_TERM_LABEL for i, term in enumerate(self.targets)}
        self._by_words = {
            " ".join(split_token(w)[1].lower() for w in t.split()): t for t in self.targets
        }
        self._longest = max((len(t.split()) for t in self.targets), default=1)

    def find(self, tokens: list[str]) -> dict[int, tuple[int, str]]:
        """Term occurrences: start index → (word count, term)."""
        found: dict[int, tuple[int, str]] = {}
        i = 0
        while i < len(tokens):
            for n in range(self._longest, 0, -1):
                key = " ".join(split_token(t)[1].lower() for t in tokens[i : i + n])
                if len(tokens[i : i + n]) == n and key in self._by_words:
                    found[i] = (n, self._by_words[key])
                    i += n
                    break
            else:
                i += 1
        return found


def label_rule_matches(
    tokens: list[str], spans: list[tuple[int, int, str]], line: str, vocab: TermVocabulary
) -> list[int]:
    """Labels for a raw line from rule matches given as (start, end, term) char spans."""
    labels: list[int] = []
    position = 0
    for token in tokens:
        start = line.index(token, position)
        end = start + len(token)
        position = end
        label = KEEP
        for span_start, span_end, term in spans:
            if span_start < end and span_end > start:
                continues_span = span_start < start and labels and labels[-1] != KEEP
                label = DELETE if continues_span else vocab.label[term]
                break
        labels.append(label)
    return labels


class ExampleBuilder:
    def __init__(
        self,
        vocab: TermVocabulary,
        find_rule_spans: Callable[[str], list[tuple[int, int, str]]],
        excluded: set[str],
        rng: random.Random,
    ) -> None:
        self._vocab = vocab
        self._find_rule_spans = find_rule_spans
        self._leaks = LeakFilter(excluded)
        self._rng = rng
        self._distorter = Distorter(rng)

    def from_raw(self, line: str) -> Example | None:
        spans = self._find_rule_spans(line)
        tokens = line.split()
        if not spans or not tokens or len(tokens) > MAX_WORDS or self._is_leak(line):
            return None
        return Example(tokens, label_rule_matches(tokens, spans, line, self._vocab))

    def from_clean(self, text: str) -> list[Example]:
        tokens = text.split()
        if not tokens or len(tokens) > MAX_WORDS or self._is_leak(text):
            return []
        examples = []
        found = self._vocab.find(tokens)
        if found:
            for _ in range(_CORRUPTIONS_PER_PHRASE):
                if example := self._corrupt(tokens, found):
                    examples.append(example)
        if self._rng.random() < _NEGATIVE_PROBABILITY:
            examples.append(Example(tokens, [KEEP] * len(tokens)))
        return examples

    def _is_leak(self, text: str) -> bool:
        return any(self._leaks.is_leak(s) for s in normalize_sentences(text))

    def _distortion(self, term: str) -> str | None:
        real = self._vocab.distortions.get(term, [])
        if not _LATIN.search(term):
            # Russian words get only distortions that really happened: synthetic typos
            # would teach the model to "fix" valid forms like «логе» → «логи»
            return self._rng.choice(real) if real else None
        if real and self._rng.random() < _REAL_DISTORTION_PROBABILITY:
            return self._rng.choice(real)
        return self._distorter.synthetic(term)

    def _corrupt(self, tokens: list[str], found: dict[int, tuple[int, str]]) -> Example | None:
        out_tokens: list[str] = []
        labels: list[int] = []
        i = 0
        while i < len(tokens):
            n, term = found.get(i, (1, ""))
            distortion = (
                self._distortion(term)
                if term and self._rng.random() < _CORRUPT_PROBABILITY
                else None
            )
            if distortion is None:
                out_tokens.extend(tokens[i : i + n])
                labels.extend([KEEP] * n)
                i += n
                continue
            words = self._distorter.recase(distortion).split() or [distortion]
            words[0] = split_token(tokens[i])[0] + words[0]
            words[-1] = words[-1] + split_token(tokens[i + n - 1])[2]
            out_tokens.extend(words)
            labels.extend([self._vocab.label[term]] + [DELETE] * (len(words) - 1))
            i += n
        if all(label == KEEP for label in labels):
            return None
        return Example(out_tokens, labels)


def chunk_phrase(text: str) -> list[str]:
    """Split long dictation into sentence-sized pieces the model can take."""
    parts: list[str] = []
    current: list[str] = []
    for token in text.split():
        current.append(token)
        if len(current) >= MAX_WORDS or token[-1:] in ".!?":
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return parts


def build_examples(
    raw_phrases: Iterable[str],
    apply_replacements: Callable[[str], str],
    verified_texts: Iterable[str],
    builder: ExampleBuilder,
) -> list[Example]:
    examples: list[Example] = []
    clean_texts: list[str] = []
    for phrase in dict.fromkeys(raw_phrases):
        for part in chunk_phrase(phrase):
            if example := builder.from_raw(part):
                examples.append(example)
        clean_texts.append(apply_replacements(phrase))
    clean_texts.extend(verified_texts)
    for text in dict.fromkeys(clean_texts):
        for part in chunk_phrase(text):
            examples.extend(builder.from_clean(part))
    return examples
