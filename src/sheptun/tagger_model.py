"""MLX model and training loop for the term tagger (see term_tagger.py).

A char-CNN turns each word's spelling into a vector, so unseen distortions ('мидлвэр')
land near known ones ('миддлвар'); a small Transformer adds the phrase context; a linear
head scores KEEP / DELETE / every term.
"""

import json
import random
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx  # type: ignore[import-untyped]
import mlx.nn as nn  # type: ignore[import-untyped]
import mlx.optimizers as optim  # type: ignore[import-untyped]
import numpy as np

from sheptun.tagger_data import Example
from sheptun.term_tagger import (
    CHARS_FILE,
    CONFIG_FILE,
    KEEP,
    MAX_CHARS,
    MAX_WORDS,
    TARGETS_FILE,
    WEIGHTS_FILE,
    CharVocab,
    Predictor,
)

_KERNELS = (2, 3, 4, 5)
_MASKED = -1e9
_IGNORE_LABEL = -1
_BATCH_WORDS = 2048
_PEAK_LEARNING_RATE = 2e-3
_FINAL_LEARNING_RATE = 1e-5
_WEIGHT_DECAY = 0.01
# KEEP is ~88% of words: without extra weight on edits the model learns to never edit
_EDIT_LOSS_WEIGHT = 2.0


@dataclass(frozen=True)
class TaggerConfig:
    n_chars: int
    n_labels: int
    dims: int = 256
    layers: int = 3
    heads: int = 4
    char_dim: int = 48
    filters: int = 96
    dropout: float = 0.1


class _WordEncoder(nn.Module):  # type: ignore[misc]
    def __init__(self, config: TaggerConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(config.n_chars, config.char_dim)
        self.convs = [nn.Conv1d(config.char_dim, config.filters, k) for k in _KERNELS]
        self.proj = nn.Linear(config.filters * len(_KERNELS), config.dims)

    def __call__(self, chars: Any) -> Any:
        batch, words, n_chars = chars.shape
        x = self.embed(chars.reshape(batch * words, n_chars))
        pooled = [nn.relu(conv(x)).max(axis=1) for conv in self.convs]
        return self.proj(mx.concatenate(pooled, axis=-1)).reshape(batch, words, -1)


class TermTaggerModel(nn.Module):  # type: ignore[misc]
    def __init__(self, config: TaggerConfig) -> None:
        super().__init__()
        self.words = _WordEncoder(config)
        self.position = nn.Embedding(MAX_WORDS, config.dims)
        self.encoder = nn.TransformerEncoder(
            config.layers,
            config.dims,
            config.heads,
            mlp_dims=config.dims * 2,
            dropout=config.dropout,
        )
        self.head = nn.Linear(config.dims, config.n_labels)

    def __call__(self, chars: Any, lengths: Any) -> Any:
        words = chars.shape[1]
        x = self.words(chars) + self.position(mx.arange(words))
        valid = mx.arange(words)[None, :] < lengths[:, None]
        mask = mx.where(valid, 0.0, _MASKED)[:, None, None, :].astype(x.dtype)
        return self.head(self.encoder(x, mask))


def _batches(
    examples: list[Example], vocab: CharVocab, rng: random.Random | None
) -> Iterator[tuple[Any, Any, Any]]:
    ordered = sorted(examples, key=lambda e: len(e.tokens))
    groups: list[list[Example]] = []
    current: list[Example] = []
    for example in ordered:
        current.append(example)
        if len(current) * len(example.tokens) >= _BATCH_WORDS:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if rng is not None:
        rng.shuffle(groups)
    for group in groups:
        width = max(len(e.tokens) for e in group)
        chars = np.zeros((len(group), width, MAX_CHARS), dtype=np.int32)
        labels = np.full((len(group), width), _IGNORE_LABEL, dtype=np.int32)
        for row, example in enumerate(group):
            for col, token in enumerate(example.tokens):
                chars[row, col] = vocab.encode(token)
            labels[row, : len(example.labels)] = example.labels
        lengths = np.array([len(e.tokens) for e in group], dtype=np.int32)
        yield mx.array(chars), mx.array(lengths), mx.array(labels)


def _loss(model: TermTaggerModel, chars: Any, lengths: Any, labels: Any) -> Any:
    logits = model(chars, lengths)
    valid = labels != _IGNORE_LABEL
    losses = nn.losses.cross_entropy(logits, mx.maximum(labels, 0), reduction="none")
    weights = mx.where(labels > KEEP, _EDIT_LOSS_WEIGHT, 1.0) * valid
    return (losses * weights).sum() / weights.sum()


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    loss: float
    precision: float
    recall: float


def _evaluate(
    model: TermTaggerModel, examples: list[Example], vocab: CharVocab
) -> tuple[float, float]:
    model.eval()
    correct = predicted = expected = 0
    for chars, lengths, labels in _batches(examples, vocab, rng=None):
        pred = np.array(mx.argmax(model(chars, lengths), axis=-1))
        gold = np.array(labels)
        valid = gold != _IGNORE_LABEL
        edits_pred = valid & (pred != KEEP)
        edits_gold = valid & (gold != KEEP)
        correct += int((edits_pred & (pred == gold)).sum())
        predicted += int(edits_pred.sum())
        expected += int(edits_gold.sum())
    model.train()
    return correct / max(predicted, 1), correct / max(expected, 1)


def train_tagger(
    train: list[Example],
    validation: list[Example],
    targets: list[str],
    output_dir: Path,
    epochs: int,
    on_epoch: Callable[[EpochReport], None],
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    mx.random.seed(seed)
    vocab = CharVocab.build(token for example in train for token in example.tokens)
    config = TaggerConfig(n_chars=len(vocab), n_labels=len(targets) + 2)
    model = TermTaggerModel(config)

    steps_per_epoch = sum(1 for _ in _batches(train, vocab, rng=None))
    schedule = optim.cosine_decay(
        _PEAK_LEARNING_RATE, epochs * steps_per_epoch, end=_FINAL_LEARNING_RATE
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=_WEIGHT_DECAY)
    step = nn.value_and_grad(model, _loss)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for chars, lengths, labels in _batches(train, vocab, rng):
            loss, grads = step(model, chars, lengths, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total += float(loss.item())
        precision, recall = _evaluate(model, validation, vocab)
        on_epoch(EpochReport(epoch, total / steps_per_epoch, precision, recall))

    _save(model, config, vocab, targets, output_dir)


def _save(
    model: TermTaggerModel,
    config: TaggerConfig,
    vocab: CharVocab,
    targets: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_dir / WEIGHTS_FILE))
    (output_dir / CONFIG_FILE).write_text(json.dumps(asdict(config)), encoding="utf-8")
    (output_dir / CHARS_FILE).write_text(
        json.dumps(vocab.chars, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / TARGETS_FILE).write_text(
        json.dumps(targets, ensure_ascii=False), encoding="utf-8"
    )


def load_predictor(model_dir: Path, n_targets: int) -> Predictor:
    config = TaggerConfig(**json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8")))
    if config.n_labels != n_targets + 2:
        raise ValueError(f"{model_dir}: модель и targets.json не совпадают, переобучите")
    vocab = CharVocab(json.loads((model_dir / CHARS_FILE).read_text(encoding="utf-8")))
    model = TermTaggerModel(config)
    model.load_weights(str(model_dir / WEIGHTS_FILE))
    model.eval()

    def predict(tokens: list[str]) -> tuple[list[int], list[float]]:
        chars = mx.array(np.array([[vocab.encode(t) for t in tokens]], dtype=np.int32))
        probs = mx.softmax(model(chars, mx.array([len(tokens)])), axis=-1)[0]
        labels = mx.argmax(probs, axis=-1)
        confidence = mx.take_along_axis(probs, labels[:, None], axis=-1)[:, 0]
        label_ids = np.asarray(labels, dtype=np.int64)
        scores = np.asarray(confidence, dtype=np.float64)
        return [int(v) for v in label_ids], [float(v) for v in scores]

    return predict
