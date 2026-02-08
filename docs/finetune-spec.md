# Fine-tuning Whisper на верифицированных данных Sheptun

## Цель

Создать модуль `src/sheptun/finetune.py` и CLI-команды для fine-tuning модели Whisper на данных из `dataset/verification.db`, чтобы улучшить распознавание русской речи в контексте управления терминалом.

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
- **evaluate** + **jiwer** — метрика WER (Word Error Rate)
- **accelerate** — поддержка GPU/MPS
- **tensorboard** — визуализация обучения

## Пайплайн

```
verification.db → Подготовка Dataset → Feature Extraction → Training → Evaluation → Export
```

### Этап 1: Подготовка данных

- Читаем из `verification.db` записи со `status='completed'` и `is_hallucination=0`
- Текст = `verified_text` (уже содержит исправленный или подтверждённый оригинал)
- Аудио = `dataset/audio/{file}`
- Фильтрация: пропускаем записи с `confidence='low'` (опционально)
- Галлюцинации (`is_hallucination=1`) исключаются автоматически — у них нет полезного текста
- Разбиение на train/eval (90%/10%)
- Формат HuggingFace Dataset: `{"audio": {"array": [...], "sampling_rate": 16000}, "sentence": "текст"}`

### Этап 2: Обучение

- Базовая модель: `openai/whisper-small` (или настраиваемая через `SHEPTUN_FINETUNE_MODEL`)
- Язык: Russian
- Ключевые гиперпараметры (настраиваемые):
  - `learning_rate`: 1e-5
  - `max_steps`: 4000
  - `batch_size`: 8-16
  - `warmup_steps`: 500
  - `gradient_checkpointing`: True
  - `fp16`: True (GPU) / False (MPS/CPU)
- Метрики: WER, CER (Character Error Rate — важно для русского)

### Этап 3: Оценка

- WER на eval split до и после fine-tuning
- Сравнение с базовой моделью

### Этап 4: Экспорт

- Сохранение в локальную директорию (`models/whisper-sheptun/`)
- Совместимость с HuggingFace pipeline для inference
- Интеграция в существующий `recognition.py` через настройку в `.env`

## CLI-команды

```bash
# Подготовить датасет из verification.db
sheptun finetune-prepare

# Запустить обучение
sheptun finetune-train --model small --steps 4000 --batch-size 8

# Оценить модель
sheptun finetune-eval

# Использовать fine-tuned модель
SHEPTUN_MODEL=models/whisper-sheptun sheptun listen
```

## Настройки `.env`

```bash
SHEPTUN_FINETUNE_MODEL=small              # Базовая модель Whisper
SHEPTUN_FINETUNE_STEPS=4000               # Количество шагов обучения
SHEPTUN_FINETUNE_BATCH_SIZE=8             # Размер батча
SHEPTUN_FINETUNE_LR=1e-5                  # Learning rate
SHEPTUN_FINETUNE_OUTPUT=models/whisper-sheptun  # Куда сохранять
SHEPTUN_FINETUNE_MIN_CONFIDENCE=medium    # Минимальный confidence из verification.db
```

## Требования к железу

| Модель | VRAM | Время обучения (~30ч данных) |
|--------|------|------------------------------|
| tiny   | 2 GB | ~1-2 часа                    |
| base   | 4 GB | ~2-3 часа                    |
| small  | 8 GB | ~3-5 часов                   |
| medium | 12 GB | ~6-10 часов                 |

Apple Silicon (MPS): работает для tiny/base/small, медленнее GPU. Для medium+ рекомендуется облачный GPU.

## Минимальный датасет

- 5+ часов верифицированного аудио для заметного улучшения
- У нас ~30 часов — более чем достаточно
- Остаток верификации можно запускать параллельно с подготовкой пайплайна

## Структура файлов

```
src/sheptun/
├── finetune.py        # Подготовка данных, обучение, оценка
├── cli.py             # + команды finetune-prepare, finetune-train, finetune-eval
├── settings.py        # + настройки SHEPTUN_FINETUNE_*
└── recognition.py     # + поддержка локального пути к модели

pyproject.toml         # + optional dependency group [finetune]
```

## Зависимости

```toml
[project.optional-dependencies]
finetune = [
    "transformers>=4.30.0",
    "datasets[audio]>=2.14.0",
    "accelerate>=0.20.0",
    "evaluate>=0.4.0",
    "jiwer>=3.0.0",
    "tensorboard>=2.14.0",
]
```

## Ожидаемые результаты

| Метрика | До fine-tuning | После fine-tuning |
|---------|---------------|-------------------|
| WER     | 30-50%        | 10-20%            |
| CER     | 15-25%        | 5-10%             |

Улучшение распознавания терминальных команд, технических терминов и смешанной русско-английской лексики.
