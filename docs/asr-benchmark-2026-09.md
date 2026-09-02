# Бенчмарк новых ASR-моделей (1 сентября 2026)

*Проверка кандидатов, появившихся после ресёрча `asr-research.md` (2 июля 2026).*
*Железо: M2 Max, macOS 13.7. Данные: `dataset/testset` — 20 фраз с эталонами (`references.jsonl`).*
*Метрика: CER норм. (без пунктуации/регистра) / CER точн. (с пунктуацией), RTF.*

## Результаты

| Модель | CER норм. | CER точн. | RTF | Где считалось |
|---|---:|---:|---:|---|
| **MLX Whisper turbo** (текущая) | **11%** | **14%** | 0.13 | GPU (MLX) |
| **GigaAM v3_e2e_ctc** | 15% | 18% | **0.05** | CPU (torch) |
| GigaAM v3_e2e_rnnt | 18% | 20% | 0.05 | CPU (torch) |
| Qwen3-ASR 1.7B | 21% | 23% | 0.30 | GPU (MLX) |
| Qwen3-ASR 0.6B | 24% | 26% | 0.09 | GPU (MLX) |
| GigaAM Multilingual 600M (`multilingual_large_ctc`) | 29% | 36% | 0.12 | CPU (torch) |
| GigaAM Multilingual 220M int8 ONNX | 30% | 37% | **0.01** | CPU (onnxruntime) |

Воспроизвести: `sheptun benchmark --testset -m mlx:turbo,qwen,gigaam -n 0`.
Для torch-моделей GigaAM использовался разовый скрипт (пакет `gigaam` из git main, после
эксперимента удалён — он откатывает torch 2.9.1 → 2.5.1).

## Выводы

### 1. GigaAM Multilingual нам не подходит

Модель вышла 14 июля 2026 (MIT, ru/en/kk/ky/uz) и по WER-таблицам авторов бьёт Whisper-large-v3
на русском (CV 5.1 против 9.1). На нашем материале — провал: **charwise CTC транслитерирует
англоязычные термины кириллицей**.

```
эталон:  git commit                          → гид комит
эталон:  ruff check mypy и pyright           → раф чек май пай и пай райт
эталон:  ветку feature-auth                  → ветку фичер дефиз алф
```

600M не лучше 220M (29% против 30%) — дело не в размере, а в природе посимвольного CTC.
Заявление автора ONNX-порта (`i2z1/gigaam-multilingual-ctc-onnx-int8`) про «латиницу для
английских терминов» на нашем тест-сете не подтвердилось.

### 2. Русская GigaAM v3 e2e — реальный кандидат

Не мультиязычная, а обычная `v3_e2e_ctc`: **15%/18% при RTF 0.05 на CPU** (Whisper turbo — 0.13
на GPU). Даёт то, что у нас числится в roadmap как «Диктовка с пунктуацией», без LLM-слоя:

```
Установи задержки на 2 секунды.       (нормализация чисел)
Запусти тест на 10 записях.
Открой fil.env и измени модель на medium.   (латиница появляется)
```

`v3_e2e_rnnt` по CER хуже (18%/20%), но чаще пишет термины латиницей (`app`, `Cloud`,
`Sheptun Bench Mac`) — вариант для доменных фраз.

### 3. Смена ASR обнуляет `replacements.yaml`

Прогон выхода GigaAM через наш пайплайн (`apply_replacements` → `TechnicalFormatter` →
`TextCleaner`) дал 30% → 28%. Наши 2074 правила заточены под галлюцинации Whisper
(`ритми.md`, `вапи`, `ров чек`) и почти не ловят фонетическую кириллицу GigaAM.
**Переезд на другой ASR = новый прогон `analyze-replacements` с нуля.**

### 4. Qwen3-ASR: 1.7B лучше 0.6B, но не догоняет

21%/23% против 24%/26%, ценой RTF 0.30 (втрое медленнее Whisper). Отдельно: у
`mlx_qwen3_asr.Session.transcribe` есть параметр `context: str` (биасинг по словарю), который мы
**не используем** — передаём только `language="Russian"`. Не проверено.

## Что осталось в коде

- `src/sheptun/gigaam_onnx.py` — бэкенд GigaAM, `SHEPTUN_RECOGNIZER=gigaam`, зарегистрирован в
  `engine.py` и `benchmark.py`. Модель выбирается `SHEPTUN_GIGAAM_MODEL` (дефолт
  `gigaam-v3-e2e-ctc`), квант — `SHEPTUN_GIGAAM_QUANTIZATION`; зависимости в extras `.[gigaam]`.
  Первая версия повторяла препроцессинг вручную — заменена на `onnx-asr` (см. «Второй заход»).
- **sherpa-onnx на macOS 13.7 не работает**: их wheel собран под macOS 26.5, встроенный
  `libonnxruntime.dylib` требует символы CoreML новее нашей ОС. Поэтому инференс — на нашем
  `onnxruntime` напрямую.

## Второй заход (2 сентября): GigaAM v3 e2e как рабочий бэкенд

Первый замер шёл через пакет `gigaam` (torch, откатывает нам версию). Переехали на
**`onnx-asr`** (`istupakov/onnx-asr` + веса `istupakov/gigaam-v3-onnx`, MIT): чистый onnxruntime,
без torch/torchaudio в горячем пути, правильный препроцессинг для каждой модели зашит внутри
(у v3 e2e он другой: `n_fft/win_length=320`, `center=false`, BPE-словарь на 257 классов).

| Вариант | CER | RTF | Размер |
|---|---:|---:|---:|
| **gigaam-v3-e2e-ctc fp32** | **15% / 18%** | 0.068 | 886 МБ |
| gigaam-v3-e2e-rnnt fp32 | 18% / 20% | 0.086 | 890 МБ |
| gigaam-v3-e2e-ctc int8 | 18% / 20% | 0.046 | 225 МБ |

- **int8-квантизация стоит 3 п.п. CER** — берём fp32, RTF всё равно вдвое лучше Whisper.
- **CoreML EP не работает на macOS 13.7**: `Error computing NN outputs` при исполнении
  CoreML-узла. Остаётся `CPUExecutionProvider` (та же семья граблей, что sherpa-onnx и Metal 3.1).
- RNN-T по CER хуже, но чаще пишет термины латиницей (`app`, `Cloud`, `Sheptun Bench Mac`).

### Обрезка тишины вредит GigaAM

`_bytes_to_float_array` перед распознаванием звал `_trim_silence`. Замер по вариантам сигнала:

| Сигнал | CER |
|---|---:|
| без trim | **15% / 18%** |
| наш путь: trim_silence | 17% / 20% |
| trim + паддинг 200/500 мс | 19% / 20% и 18% / 19% |

Паддинг тишины обрезку не компенсирует. **Whisper к обрезке безразличен** (11%/14% с ней и без),
поэтому `_bytes_to_float_array` получил параметр `trim`, и GigaAM зовёт его с `trim=False`.

### Пост-пайплайн под GigaAM не настроен

Прогон выхода `v3_e2e_ctc` через `apply_replacements → TechnicalFormatter → TextCleaner`:
**15%/18% → 18%/20%**, то есть наши правила его портят. Часть случаев — конфликт правила с
эталоном (`энтер`→`Enter` против эталонного «Энтер.», `комит`→`commit` против «коммит»), часть —
правила под ошибки Whisper. Вывод прежний: переезд требует своего прогона `analyze-replacements`.

## Не проверено (следующая сессия)

1. **GigaAM v3 e2e через MLX** — ONNX-бэкенд считает на CPU (RTF 0.07); MLX-порты
   (`aystream/GigaAM-v3-e2e-ctc-mlx`, `al-bo/gigaam-v3-rnnt-mlx`) должны дать GPU.
2. **Voxtral Mini 4B Realtime** (Apache-2.0, ru, стриминг <500 мс) — через `mlx-audio`,
   ставить в отдельный venv, чтобы не тронуть наш mlx 0.29.3.
3. **VibeVoice-ASR** (Microsoft, MIT, 50+ языков, нативный code-switching) — `mlx-community`
   порты 4/6/8-bit, 4-bit весит 5.7 ГБ.
4. **`context` в Qwen3-ASR** — топ-N ключей `replacements.yaml` как биасинг перед ASR.

## VoiceInk 2.x (ориентир, релизы после 2 июля)

- **v2.0 (16.07)** — Modes: профили под приложение/сайт/задачу + trigger words (= наши
  «Контекстные команды» из roadmap). Плюс Assistant, статистика.
- **v2.11 (12.08)** — **VoiceInk Refine**, локальная очистка транскрипта. По коду:
  `beingpax/VoiceInk-Refine-V1` — файнтюн **Qwen3.5-2B**, 4-bit MLX, ~1.06 ГБ, отдельный
  **XPC-процесс**, temperature 0.3, системный промпт в одну фразу, требует Apple Silicon + 16 ГБ.
  **Только английский; лицензия запрещает использование вне VoiceInk** — берём рецепт, не модель.
- **v2.13 (27.08)** — Gemini 3.5 Transcribe, SenseVoice Small.
- Локальные бэкенды: whisper.cpp + GGUF-каталог + FluidAudio (Parakeet unified, Nemotron
  multilingual streaming через CoreML/ANE).

**Вывод для нашего LLM-enhancement:** блокер решается не «моделью побольше» (Qwen2.5-7B
галлюцинировал), а **task-specific файнтюном ~2B** — у нас для этого есть `finetune.py`,
`dataset/verification.db` и словарь замен как источник пар.
