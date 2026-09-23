"""Term tagger: fixes ASR-mangled terms after recognition ('мидлвэр' → 'middleware').

Every word gets KEEP, DELETE (swallowed by the replacement before it) or a term from a
closed list, so the model cannot invent text. An edit is applied only when the model is
confident enough. The MLX model lives in `tagger_model.py` and is imported lazily.
"""

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("sheptun")

KEEP, DELETE = 0, 1
FIRST_TERM_LABEL = 2
MAX_CHARS = 24
MAX_WORDS = 64
PAD_CHAR, UNKNOWN_CHAR = 0, 1
_FIRST_CHAR_ID = 2
_PUNCTUATION = ".,!?;:«»\"'()[]…—–"

CHARS_FILE = "chars.json"
TARGETS_FILE = "targets.json"
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "weights.safetensors"

Predictor = Callable[[list[str]], tuple[list[int], list[float]]]


def split_token(token: str) -> tuple[str, str, str]:
    """'«Commit,' → ('«', 'Commit', ',')."""
    core = token.strip(_PUNCTUATION)
    if not core:
        return token, "", ""
    start = token.index(core)
    return token[:start], core, token[start + len(core) :]


class CharVocab:
    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self._index = {ch: i + _FIRST_CHAR_ID for i, ch in enumerate(chars)}

    @classmethod
    def build(cls, tokens: Iterable[str], min_count: int = 3) -> "CharVocab":
        counts: dict[str, int] = {}
        for token in tokens:
            for ch in token:
                counts[ch] = counts.get(ch, 0) + 1
        return cls(sorted(ch for ch, n in counts.items() if n >= min_count))

    def __len__(self) -> int:
        return len(self.chars) + _FIRST_CHAR_ID

    def encode(self, token: str) -> list[int]:
        ids = [self._index.get(ch, UNKNOWN_CHAR) for ch in token[:MAX_CHARS]]
        return ids + [PAD_CHAR] * (MAX_CHARS - len(ids))


def apply_labels(
    tokens: list[str],
    labels: list[int],
    confidence: list[float],
    targets: list[str],
    threshold: float,
) -> str:
    """Rebuild the phrase: confident replacements swallow the DELETE words after them."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        label = labels[i]
        if label < FIRST_TERM_LABEL or confidence[i] < threshold:
            out.append(tokens[i])
            i += 1
            continue
        end = i + 1
        while end < len(tokens) and labels[end] == DELETE and confidence[end] >= threshold:
            end += 1
        prefix = split_token(tokens[i])[0]
        suffix = split_token(tokens[end - 1])[2]
        out.append(prefix + targets[label - FIRST_TERM_LABEL] + suffix)
        i = end
    return " ".join(out)


@dataclass(frozen=True)
class TermCorrector:
    predict: Predictor
    targets: list[str]
    threshold: float

    def correct(self, text: str) -> str:
        tokens = text.split()
        if not tokens or len(tokens) > MAX_WORDS:
            return text
        labels, confidence = self.predict(tokens)
        return apply_labels(tokens, labels, confidence, self.targets, self.threshold)


def load_term_corrector(model_dir: Path, threshold: float) -> TermCorrector:
    from sheptun.tagger_model import load_predictor

    targets: list[str] = json.loads((model_dir / TARGETS_FILE).read_text(encoding="utf-8"))
    return TermCorrector(load_predictor(model_dir, len(targets)), targets, threshold)


def create_term_corrector(model_path: str | None, threshold: float) -> TermCorrector | None:
    """Corrector from SHEPTUN_TAGGER_PATH, or None when not configured or not trained yet."""
    if not model_path:
        return None
    model_dir = Path(model_path).expanduser()
    if not (model_dir / WEIGHTS_FILE).exists():
        logger.warning(f"Term tagger не найден: {model_dir} (обучите: sheptun train-tagger)")
        return None
    logger.info(f"Term tagger: {model_dir}, порог {threshold}")
    return load_term_corrector(model_dir, threshold)
