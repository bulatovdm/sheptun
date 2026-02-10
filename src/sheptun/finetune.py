from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("sheptun")

WHISPER_MODELS: dict[str, str] = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
}

CONFIDENCE_LEVELS: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def resolve_model_id(model_name: str) -> str:
    """Short name ('large') → HuggingFace model ID, or pass through."""
    return WHISPER_MODELS.get(model_name, model_name)


@dataclass(frozen=True)
class FinetuneConfig:
    base_model: str
    method: str
    output_dir: Path
    max_steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    min_confidence: str
    dataset_path: Path
    eval_split: float = 0.1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25
    gradient_checkpointing: bool = True


def config_from_settings(**overrides: Any) -> FinetuneConfig:
    from sheptun.settings import settings

    return FinetuneConfig(
        base_model=resolve_model_id(overrides.get("model") or settings.finetune_model),
        method=overrides.get("method") or settings.finetune_method,
        output_dir=overrides.get("output") or settings.finetune_output,
        max_steps=overrides.get("steps") or settings.finetune_steps,
        batch_size=overrides.get("batch_size") or settings.finetune_batch_size,
        learning_rate=overrides.get("lr") or settings.finetune_lr,
        warmup_steps=overrides.get("warmup_steps") or settings.finetune_warmup_steps,
        min_confidence=overrides.get("min_confidence") or settings.finetune_min_confidence,
        dataset_path=overrides.get("dataset") or settings.dataset_path,
    )


def _allowed_confidences(min_confidence: str) -> list[str]:
    min_level = CONFIDENCE_LEVELS.get(min_confidence, 1)
    return [name for name, level in CONFIDENCE_LEVELS.items() if level >= min_level]


def _load_records(config: FinetuneConfig) -> list[dict[str, str]]:
    db_path = config.dataset_path / "verification.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Verification DB not found: {db_path}")

    allowed = _allowed_confidences(config.min_confidence)
    placeholders = ",".join("?" for _ in allowed)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT file, verified_text FROM verifications "
            f"WHERE status='completed' AND is_hallucination=0 "
            f"AND confidence IN ({placeholders})",
            allowed,
        ).fetchall()
    finally:
        conn.close()

    audio_dir = config.dataset_path / "audio"
    records: list[dict[str, str]] = []
    skipped = 0
    for row in rows:
        text = row["verified_text"]
        if not text or not text.strip():
            skipped += 1
            continue

        audio_path = audio_dir / row["file"]
        if not audio_path.exists():
            skipped += 1
            continue

        records.append({"audio": str(audio_path), "sentence": text.strip()})

    if skipped:
        logger.info(f"Skipped {skipped} records (missing audio or empty text)")

    return records


def _preprocess_split(hf_split: Any, processor: Any, output_dir: Path, split_name: str) -> int:
    """Extract mel features + tokenize labels, save as .npz files in float16."""
    from pathlib import Path as _Path

    import librosa
    import numpy as np

    features_dir = _Path(output_dir) / "features" / split_name
    features_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(hf_split)):
        sample = hf_split[idx]
        audio, _ = librosa.load(sample["audio_path"], sr=16000, mono=True)
        features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
        labels = np.array(processor.tokenizer(sample["sentence"]).input_ids, dtype=np.int32)
        np.savez(
            features_dir / f"{idx}.npz",
            input_features=features.input_features[0].astype(np.float16),
            labels=labels,
        )
        if (idx + 1) % 500 == 0:
            logger.info(f"  {split_name}: {idx + 1}/{len(hf_split)}")

    logger.info(f"  {split_name}: {len(hf_split)}/{len(hf_split)} done")
    return len(hf_split)


def prepare_dataset(config: FinetuneConfig) -> dict[str, int]:
    """Подготовить датасет для fine-tuning из verification.db (пути + тексты)."""
    import datasets

    Dataset: Any = getattr(datasets, "Dataset")  # noqa: B009
    DatasetDict: Any = getattr(datasets, "DatasetDict")  # noqa: B009

    records = _load_records(config)
    if not records:
        raise ValueError("No records found matching the criteria")

    dataset: Any = Dataset.from_dict(
        {"audio_path": [r["audio"] for r in records], "sentence": [r["sentence"] for r in records]}
    )

    split: Any = dataset.train_test_split(test_size=config.eval_split, seed=42)
    ds: Any = DatasetDict({"train": split["train"], "test": split["test"]})

    save_path = config.output_dir / "dataset"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_path))

    logger.info("Preprocessing mel features (float16)...")
    processor: Any = _load_processor(config.base_model)
    _preprocess_split(ds["train"], processor, config.output_dir, "train")
    _preprocess_split(ds["test"], processor, config.output_dir, "test")

    return {"total": len(records), "train": len(ds["train"]), "eval": len(ds["test"])}


def _load_dataset_from_disk(path: str) -> Any:
    import datasets

    cls: Any = getattr(datasets, "DatasetDict")  # noqa: B009
    return cls.load_from_disk(path)


def _load_processor(model_name: str) -> Any:
    import transformers

    cls: Any = getattr(transformers, "WhisperProcessor")  # noqa: B009
    return cls.from_pretrained(model_name)


def _load_model(model_name: str, **kwargs: Any) -> Any:
    import transformers

    cls: Any = getattr(transformers, "WhisperForConditionalGeneration")  # noqa: B009
    return cls.from_pretrained(model_name, **kwargs)


def _get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_training_args(config: FinetuneConfig, device: str) -> Any:
    from transformers import Seq2SeqTrainingArguments

    use_bf16 = device in ("mps", "cuda")

    return Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=max(1, 16 // config.batch_size),
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_pin_memory=(device == "cuda"),
        optim="adamw_torch",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
    )


def _get_lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )


@dataclass
class _DataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return dict(batch)


def _extract_metric(result: dict[str, float] | float | None) -> float:
    if result is None:
        return 0.0
    if isinstance(result, dict):
        return float(next(iter(result.values())))
    return float(result)


def _load_metric(name: str) -> Any:
    import evaluate

    load_fn: Any = getattr(evaluate, "load")  # noqa: B009
    return load_fn(name)


class _PreprocessedDataset:
    """Torch-compatible dataset that reads pre-extracted mel features from disk."""

    def __init__(self, features_dir: Path) -> None:
        from pathlib import Path as _Path

        self._features_dir = _Path(features_dir)
        self._length = len(list(self._features_dir.glob("*.npz")))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import numpy as np
        import torch

        from_numpy: Any = getattr(torch, "from_numpy")  # noqa: B009
        data: Any = np.load(self._features_dir / f"{idx}.npz")
        features: Any = data["input_features"].astype(np.float32)
        return {
            "input_features": from_numpy(features),
            "labels": data["labels"].tolist(),
        }


class _LazyAudioDataset:
    """Torch-compatible dataset that loads audio on-the-fly (fallback)."""

    def __init__(self, hf_dataset: Any, processor: Any) -> None:
        self._dataset = hf_dataset
        self._processor = processor

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import librosa

        sample = self._dataset[idx]
        audio, _ = librosa.load(sample["audio_path"], sr=16000, mono=True)
        features = self._processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        labels = self._processor.tokenizer(sample["sentence"]).input_ids
        return {
            "input_features": features.input_features[0],
            "labels": labels,
        }


def _make_compute_metrics(processor: Any) -> Any:
    wer_metric = _load_metric("wer")
    cer_metric = _load_metric("cer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "wer": _extract_metric(wer_metric.compute(predictions=pred_str, references=label_str)),
            "cer": _extract_metric(cer_metric.compute(predictions=pred_str, references=label_str)),
        }

    return compute_metrics


def _create_asr_pipeline(
    model: Any,
    device: str,
    tokenizer: Any = None,
    feature_extractor: Any = None,
) -> Any:
    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "model": model,
        "device": device,
        "generate_kwargs": {"language": "russian", "task": "transcribe"},
    }
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    return pipeline("automatic-speech-recognition", **kwargs)


def train(config: FinetuneConfig, resume: bool = False) -> Path:
    """Запустить fine-tuning модели Whisper."""
    import torch
    import transformers

    Seq2SeqTrainer: Any = getattr(transformers, "Seq2SeqTrainer")  # noqa: B009

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    device = _get_device()
    logger.info(f"Training on device: {device}")

    dataset_path = config.output_dir / "dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run 'sheptun finetune-prepare' first."
        )
    ds: Any = _load_dataset_from_disk(str(dataset_path))

    processor: Any = _load_processor(config.base_model)
    model: Any = _load_model(
        config.base_model,
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    model.generation_config.language = "russian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model.config.use_cache = False

    if config.method == "lora":
        from peft import get_peft_model

        model.enable_input_require_grads()
        lora_config = _get_lora_config()
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({trainable / total:.2%})")

    features_path = config.output_dir / "features"
    if (features_path / "train").exists():
        logger.info("Using preprocessed mel features")
        train_ds: Any = _PreprocessedDataset(features_path / "train")
        eval_ds: Any = _PreprocessedDataset(features_path / "test")
    else:
        logger.info("No preprocessed features, using lazy audio loading (slower)")
        train_ds = _LazyAudioDataset(ds["train"], processor)
        eval_ds = _LazyAudioDataset(ds["test"], processor)

    training_args = _get_training_args(config, device)
    data_collator = _DataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer: Any = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    checkpoint = None
    if resume:
        checkpoints = sorted((config.output_dir / "checkpoints").glob("checkpoint-*"))
        if checkpoints:
            checkpoint = str(checkpoints[-1])

    trainer.train(resume_from_checkpoint=checkpoint)

    output_path = config.output_dir
    if config.method == "lora":
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_path))
    else:
        model.save_pretrained(str(output_path))

    processor.save_pretrained(str(output_path))
    logger.info(f"Model saved to {output_path}")

    return output_path


def _run_predictions(pipe: Any, dataset: Any) -> list[str]:
    import librosa

    preds: list[str] = []
    for sample in dataset:
        audio, _ = librosa.load(sample["audio_path"], sr=16000, mono=True)
        result = pipe({"raw": audio, "sampling_rate": 16000}, return_timestamps=False)
        preds.append(result["text"].strip())
    return preds


def evaluate(config: FinetuneConfig) -> dict[str, float]:
    """Оценить fine-tuned модель vs базовую (WER/CER)."""
    dataset_path = config.output_dir / "dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ds: Any = _load_dataset_from_disk(str(dataset_path))
    eval_ds: Any = ds["test"]

    device = _get_device()
    wer_metric = _load_metric("wer")
    cer_metric = _load_metric("cer")

    references: list[str] = eval_ds["sentence"]

    results: dict[str, float] = {}

    base_pipe = _create_asr_pipeline(model=config.base_model, device=device)
    base_preds = _run_predictions(base_pipe, eval_ds)
    results["wer_base"] = _extract_metric(
        wer_metric.compute(predictions=base_preds, references=references)
    )
    results["cer_base"] = _extract_metric(
        cer_metric.compute(predictions=base_preds, references=references)
    )
    del base_pipe

    model_path = str(config.output_dir)
    ft_pipe = _create_asr_pipeline(model=model_path, device=device)
    ft_preds = _run_predictions(ft_pipe, eval_ds)
    results["wer_finetuned"] = _extract_metric(
        wer_metric.compute(predictions=ft_preds, references=references)
    )
    results["cer_finetuned"] = _extract_metric(
        cer_metric.compute(predictions=ft_preds, references=references)
    )

    return results
