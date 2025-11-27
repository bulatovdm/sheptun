# Персонализация Whisper под голос пользователя

Исследовательский документ по дообучению модели распознавания речи.

## Обзор подходов

### 1. Fine-tuning (полное дообучение)

**Что это:** Дообучение всех весов модели Whisper на новых данных.

**Требования:**
- GPU с 24+ GB VRAM (для large модели)
- 8-12 часов аудио с транскрипцией
- Время обучения: несколько часов на мощном GPU

**Плюсы:**
- Максимальное качество адаптации
- Модель полностью подстраивается под голос

**Минусы:**
- Требует много данных и ресурсов
- Риск "забывания" общих знаний (catastrophic forgetting)
- Большие checkpoint файлы (~7GB)

### 2. LoRA / PEFT (эффективное дообучение) ⭐ Рекомендуется

**Что это:** Low-Rank Adaptation — обучаются только маленькие матрицы-адаптеры, основные веса заморожены.

**Требования:**
- GPU с 8+ GB VRAM (можно на consumer GPU)
- 1-8 часов аудио с транскрипцией
- Время обучения: 1-2 часа

**Плюсы:**
- Работает на обычных GPU (RTX 3080, MacBook M1/M2)
- Маленькие checkpoint файлы (~60MB вместо 7GB)
- Меньше риск переобучения
- Можно комбинировать несколько LoRA адаптеров

**Минусы:**
- Немного уступает полному fine-tuning по качеству

**Конфигурация LoRA:**
```python
from peft import LoraConfig

config = LoraConfig(
    r=32,                          # rank матриц
    lora_alpha=64,                 # scaling factor
    target_modules=["q_proj", "v_proj"],  # какие слои адаптировать
    lora_dropout=0.05,
    bias="none"
)
```

### 3. Speech-based In-Context Learning (SICL)

**Что это:** Адаптация на этапе инференса без изменения весов модели.

**Требования:**
- Несколько примеров (few-shot)
- Не требует обучения

**Результаты исследований:**
- Снижение WER на 32-36% для диалектов
- Сравнимо с LoRA без gradient descent

**Плюсы:**
- Не требует обучения
- Мгновенная адаптация

**Минусы:**
- Ограниченные улучшения
- Увеличивает latency

### 4. Пользовательский словарь (post-processing)

**Что это:** Автозамена частых ошибок распознавания.

**Пример:**
```yaml
corrections:
  "клот": "клод"
  "питон": "python"
  "жить": "git"
```

**Плюсы:**
- Не требует обучения
- Мгновенный эффект
- Легко обновлять

**Минусы:**
- Только исправляет известные ошибки
- Не улучшает само распознавание

---

## Рекомендуемый план действий

### Этап 1: Пользовательский словарь (сразу)
**Цель:** Быстрое исправление частых ошибок

1. Собрать список частых ошибок распознавания
2. Создать файл коррекций `~/.config/sheptun/corrections.yaml`
3. Применять post-processing после распознавания

**Формат данных:**
```yaml
corrections:
  "клот": "клод"
  "код": "code"
```

### Этап 2: Сбор датасета (1-2 недели)
**Цель:** Накопить аудио для обучения

1. Добавить команду `sheptun record-dataset`
2. Записывать аудио + показывать что распознано
3. Пользователь подтверждает или исправляет транскрипцию
4. Сохранять пары (аудио, текст) в директорию

**Требования к данным:**
- Формат: WAV, 16kHz, mono
- Длина клипов: 5-30 секунд
- Цель: минимум 1-2 часа для LoRA, 8+ часов для full fine-tuning

**Структура датасета:**
```
~/.config/sheptun/dataset/
├── audio/
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
├── transcripts.txt      # формат: filename|transcript
└── metadata.json        # статистика
```

**Формат transcripts.txt:**
```
001.wav|клод напиши функцию
002.wav|таб энтер
003.wav|слэш хелп
```

### Этап 3: LoRA Fine-tuning (когда есть 1+ час данных)
**Цель:** Дообучить модель на голос пользователя

1. Подготовить датасет в формате HuggingFace
2. Запустить обучение с LoRA
3. Сохранить адаптер (~60MB)
4. Интегрировать в Sheptun

**Скрипт обучения:**
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

# Загрузить базовую модель
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

# Конфигурация LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Обернуть модель
model = get_peft_model(model, lora_config)

# Обучение...
```

### Этап 4: Оценка и итерация
**Цель:** Измерить улучшение и продолжить сбор данных

1. Измерить WER до и после
2. Найти оставшиеся проблемные слова
3. Добавить больше примеров для проблемных случаев
4. Повторить обучение

---

## Технические требования

### Для сбора данных
- Микрофон (уже есть)
- ~1GB свободного места на час аудио

### Для LoRA обучения
- **Минимум:** GPU 8GB VRAM или MacBook M1/M2 с 16GB RAM
- **Рекомендуется:** GPU 16GB+ VRAM (RTX 4080, A100)
- Python библиотеки: `transformers`, `peft`, `datasets`, `accelerate`

### Для full fine-tuning
- GPU 24GB+ VRAM
- Или использовать облако (Google Colab Pro, AWS SageMaker)

### Mac Studio M2 Max (32GB) ✅
**Этой конфигурации достаточно для всех задач:**
- LoRA fine-tuning whisper-large — ✅ без проблем
- Full fine-tuning whisper-medium — ✅ возможно
- Full fine-tuning whisper-large — ⚠️ на пределе

PyTorch использует MPS (Metal Performance Shaders) backend:
```python
device = "mps"  # вместо "cuda"
```

Облако не требуется.

---

## Метрики качества

### Word Error Rate (WER)
Основная метрика для ASR. Формула:
```
WER = (S + D + I) / N
```
- S = substitutions (замены)
- D = deletions (пропуски)
- I = insertions (вставки)
- N = total words (всего слов)

**Целевые показатели:**
- Baseline Whisper: ~10-15% WER на русском
- После LoRA: ~5-8% WER на голосе пользователя
- После full fine-tuning: ~3-5% WER

### Command Accuracy
Процент правильно распознанных команд:
```
Accuracy = correct_commands / total_commands
```

---

## Ресурсы

### Официальные руководства
- [Fine-Tune Whisper for Multilingual ASR](https://huggingface.co/blog/fine-tune-whisper) — HuggingFace
- [HuggingFace Audio Course](https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning) — пошаговый курс
- [AWS: Fine-tune Whisper with LoRA](https://aws.amazon.com/blogs/machine-learning/fine-tune-whisper-models-on-amazon-sagemaker-with-lora/)

### GitHub репозитории
- [fast-whisper-finetuning](https://github.com/Vaibhavs10/fast-whisper-finetuning) — notebook для consumer GPU
- [whisper-finetune](https://github.com/vasistalodagala/whisper-finetune) — скрипты для custom datasets
- [finetune-whisper-lora](https://github.com/fengredrum/finetune-whisper-lora) — PEFT + Transformers

### Исследования
- [Whisper Paper](https://cdn.openai.com/papers/whisper.pdf) — оригинальная статья OpenAI
- [Speech-based In-Context Learning](https://arxiv.org/html/2309.07081v2) — few-shot адаптация
- [S2-LoRA for Child Speech](https://arxiv.org/abs/2309.11756) — LoRA для low-resource

---

## Следующие шаги для Sheptun

1. **[Просто]** Добавить пользовательский словарь коррекций
2. **[Средне]** Команда `sheptun record-dataset` для сбора данных
3. **[Средне]** Интерфейс разметки записей
4. **[Сложно]** Интеграция LoRA обучения и инференса
