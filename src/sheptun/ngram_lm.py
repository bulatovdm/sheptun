"""KenLM n-gram language model for GigaAM beam search (shallow fusion).

The decoder emits words with casing and punctuation ("Commit."), the LM is trained on
normalized words ("commit"), so scoring goes through the same ``normalize_word``.
Training needs KenLM's ``lmplz`` and ``build_binary`` binaries (built from source).
"""

import subprocess
from pathlib import Path
from typing import Any

from pyctcdecode.language_model import LanguageModel  # type: ignore[import-untyped]

from sheptun.lm_corpus import normalize_word

_LEADING_PUNCTUATION = ".,!?;:«»\"'()—–-"
_UNIGRAMS_SUFFIX = ".unigrams.txt"


def unigrams_path(model_path: Path) -> Path:
    return model_path.with_suffix(_UNIGRAMS_SUFFIX)


class NormalizedLanguageModel(LanguageModel):  # type: ignore[misc]
    def score_partial_token(self, partial_token: str) -> float:
        normalized = partial_token.lstrip(_LEADING_PUNCTUATION).lower().replace("ё", "е")
        return float(super().score_partial_token(normalized))

    def score(self, prev_state: Any, word: str, is_last_word: bool = False) -> tuple[float, Any]:
        normalized = normalize_word(word)
        if not normalized:
            return 0.0, prev_state
        score, state = super().score(prev_state, normalized, is_last_word)
        return float(score), state


def load_language_model(
    model_path: Path, alpha: float, beta: float, unk_score_offset: float
) -> NormalizedLanguageModel:
    import kenlm  # type: ignore[import-not-found]

    unigrams = unigrams_path(model_path).read_text(encoding="utf-8").split()
    return NormalizedLanguageModel(
        kenlm.Model(str(model_path)),
        unigrams,
        alpha=alpha,
        beta=beta,
        unk_score_offset=unk_score_offset,
    )


def train_language_model(
    corpus: list[str], output: Path, order: int, lmplz: Path, build_binary: Path
) -> None:
    """ARPA via lmplz → binary via build_binary, plus the unigram list the decoder needs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    arpa = output.with_suffix(".arpa")
    with arpa.open("w", encoding="utf-8") as arpa_file:
        subprocess.run(
            [str(lmplz), "-o", str(order), "--discount_fallback", "-S", "20%"],
            input="\n".join(corpus) + "\n",
            stdout=arpa_file,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    subprocess.run([str(build_binary), str(arpa), str(output)], capture_output=True, check=True)
    arpa.unlink()
    vocabulary = sorted({word for sentence in corpus for word in sentence.split()})
    unigrams_path(output).write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
