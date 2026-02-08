# Fine-tuning Whisper на верифицированных данных Sheptun

## Цель

Модуль `src/sheptun/finetune.py` и CLI-команды для fine-tuning модели Whisper на данных из `dataset/verification.db`, чтобы улучшить распознавание русской речи в контексте управления терминалом.

## Исходные данные

- **8,652 аудиозаписи** (WAV, 16kHz, mono, 16-bit) — ~30 часов
- **Верифицированные транскрипции** в `dataset/verification.db`:
  - `file` — имя WAV файла
  - `original_text` — транскрипция Whisper
  - `verified_text` — исправленная транскрипция от Claude
  - `is_correct` — менять ли текст
  - `is_hallucination` — весь текст является галлюцинацией Whisper (мусор, повторы, чужие скрипты)
  - `confidence` — high / medium / low
- Для fine-tuning используем: аудио + `verified_text` (если есть) или `original_text` (если `is_correct=1`)
- **Исключаем** записи с `is_hallucination=1` — это мусорные транскрипции без полезного текста

## Технологический стек

- **HuggingFace Transformers** — `WhisperForConditionalGeneration`, `WhisperProcessor`, `Seq2SeqTrainer`
- **datasets[audio]** — загрузка и предобработка аудио
- **PEFT** — LoRA адаптеры для эффективного fine-tuning
- **evaluate** + **jiwer** — метрики WER (Word Error Rate) и CER (Character Error Rate)
- **accelerate** — поддержка GPU/MPS
- **tensorboard** — визуализация обучения

## Два метода обучения

### LoRA/PEFT (рекомендуется)

Обучение лёгких адаптеров поверх замороженной модели. ~0.4% параметров обучаются.

- `LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"])`
- Меньше памяти (~14 GB для large)
- Быстрее обучение
- Идеально для Apple Silicon

### Full Fine-tuning

Обновление всех весов модели. Максимальное качество, но больше памяти.

- ~20 GB для large модели
- Требует gradient checkpointing
- На грани возможностей M2 Max 32GB

## Пайплайн

```
verification.db → Подготовка Dataset → Feature Extraction → Training → Evaluation → Export
```

### Этап 1: Подготовка данных (`sheptun finetune-prepare`)

- Читаем из `verification.db` записи со `status='completed'` и `is_hallucination=0`
- Текст = `verified_text` (уже содержит исправленный или подтверждённый оригинал)
- Аудио = `dataset/audio/{file}`
- Фильтрация по `confidence` (настраивается через `SHEPTUN_FINETUNE_MIN_CONFIDENCE`)
- Галлюцинации (`is_hallucination=1`) исключаются автоматически
- Разбиение на train/eval (90%/10%, seed=42)
- Формат HuggingFace Dataset: `{"audio": Audio(16kHz), "sentence": "текст"}`

### Этап 2: Обучение (`sheptun finetune-train`)

- Базовая модель: `openai/whisper-large-v3` (настраивается через `SHEPTUN_FINETUNE_MODEL`)
- Язык: Russian
- Ключевые гиперпараметры:
  - `learning_rate`: 1e-5
  - `max_steps`: 4000
  - `batch_size`: 4 (Apple Silicon) / 8-16 (CUDA)
  - `warmup_steps`: 500
  - `gradient_accumulation_steps`: 4 (effective batch = 16)
  - `gradient_checkpointing`: True
  - `bf16`: True (MPS/CUDA) / `fp16`: False (MPS не поддерживает)
- После LoRA обучения — merge адаптеров в базовую модель
- Метрики: WER, CER

### Этап 3: Оценка (`sheptun finetune-eval`)

- WER/CER на eval split: base vs fine-tuned
- Использует HuggingFace pipeline для обеих моделей

### Этап 4: Использование

- `SHEPTUN_MODEL=models/whisper-sheptun sheptun listen`
- Автоматически определяет локальную модель и использует HuggingFace pipeline

## CLI-команды

```bash
# Установить зависимости
pip install -e ".[finetune]"

# Подготовить датасет из verification.db
sheptun finetune-prepare

# Запустить обучение (LoRA по умолчанию)
sheptun finetune-train

# Запустить обучение (полный fine-tuning)
sheptun finetune-train --method full

# С настройками
sheptun finetune-train --model large --steps 4000 --batch-size 4 --method lora

# Продолжить обучение с checkpoint
sheptun finetune-train --resume

# Оценить модель
sheptun finetune-eval

# Использовать fine-tuned модель
SHEPTUN_MODEL=models/whisper-sheptun sheptun listen
```

## Настройки `.env`

```bash
SHEPTUN_FINETUNE_MODEL=large             # Базовая модель Whisper (tiny/base/small/medium/large)
SHEPTUN_FINETUNE_METHOD=lora             # Метод обучения (lora, full)
SHEPTUN_FINETUNE_STEPS=4000              # Количество шагов обучения
SHEPTUN_FINETUNE_BATCH_SIZE=4            # Размер батча
SHEPTUN_FINETUNE_LR=1e-5                 # Learning rate
SHEPTUN_FINETUNE_WARMUP_STEPS=500        # Шаги warmup
SHEPTUN_FINETUNE_OUTPUT=models/whisper-sheptun  # Куда сохранять
SHEPTUN_FINETUNE_MIN_CONFIDENCE=medium   # Минимальный confidence из verification.db
```

## MPS-специфика (Apple Silicon)

- `fp16=False` — MPS не поддерживает fp16 для backward pass
- `bf16=True` — поддерживается на Apple Silicon (PyTorch 2.0+)
- `dataloader_pin_memory=False` — pin_memory только для CUDA
- `optim="adamw_torch"` — native PyTorch оптимизатор (стабильнее на MPS)
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — снять лимит памяти MPS
- `gradient_checkpointing=True` — обязательно для large модели

## Требования к железу

| Метод | Модель | VRAM | Время обучения (~30ч данных) |
|-------|--------|------|------------------------------|
| LoRA  | large  | ~14 GB | ~4-6 часов                |
| Full  | large  | ~20 GB | ~8-12 часов               |
| LoRA  | medium | ~10 GB | ~3-4 часов                |
| Full  | medium | ~14 GB | ~6-8 часов                |
| LoRA  | small  | ~6 GB  | ~1-2 часа                 |

Apple Silicon M2 Max (32GB): LoRA large — комфортно, Full large — на пределе.

## Минимальный датасет

- 5+ часов верифицированного аудио для заметного улучшения
- У нас ~30 часов — более чем достаточно

## Структура файлов

```
src/sheptun/
├── finetune.py        # Подготовка данных, обучение, оценка
├── cli.py             # + команды finetune-prepare, finetune-train, finetune-eval
├── settings.py        # + настройки SHEPTUN_FINETUNE_*
├── recognition.py     # + HuggingFaceWhisperRecognizer для inference
└── engine.py          # + авто-выбор recognizer по типу модели

models/whisper-sheptun/  # Сохранённая fine-tuned модель
├── config.json
├── model.safetensors
├── preprocessor_config.json
├── tokenizer.json
├── dataset/             # Подготовленный HuggingFace dataset
└── checkpoints/         # Training checkpoints
```

## Зависимости

```toml
[project.optional-dependencies]
finetune = [
    "transformers>=4.36.0",
    "datasets>=2.14.0",
    "accelerate>=0.25.0",
    "evaluate>=0.4.0",
    "jiwer>=3.0.0",
    "tensorboard>=2.14.0",
    "peft>=0.7.0",
    "soundfile>=0.12.0",
    "librosa>=0.10.0",
]
```

## Ожидаемые результаты

| Метрика | До fine-tuning | После fine-tuning |
|---------|---------------|-------------------|
| WER     | 30-50%        | 10-20%            |
| CER     | 15-25%        | 5-10%             |

Улучшение распознавания терминальных команд, технических терминов и смешанной русско-английской лексики.
