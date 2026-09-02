# Выбор ASR-модели (замеры на тест-сете)

## Полный бенчмарк 2026-09-01 — `docs/asr-benchmark-2026-09.md`
Прогон новых кандидатов на `dataset/testset` (20 фраз): Whisper turbo остаётся лидером
(CER 11%/14%, RTF 0.13). Таблица, тексты ошибок и планы — в документе. Команда:
`sheptun benchmark --testset -m mlx:turbo,qwen,gigaam -n 0`.

## GigaAM Multilingual (июль 2026) НЕ подходит — транслитерирует термины
Несмотря на лучшие в классе WER для русского в карточке модели, charwise CTC пишет англоязычные
термины кириллицей: `git commit`→«гид комит», `ruff check mypy`→«раф чек май пай». CER 30%/37%
(220M int8) и 29%/36% (600M) против 11%/14% у Whisper. Размер не решает — решает природа
посимвольного CTC. Заявление ONNX-порта `i2z1/...` про «латиницу для терминов» не подтвердилось.
**Не возвращаться к мультиязычной линии GigaAM без нового доказательства.**

## Бэкенд GigaAM: `gigaam.py`, два рантайма, MLX по умолчанию
`SHEPTUN_RECOGNIZER=gigaam` → `src/sheptun/gigaam.py`. Рантайм выбирается
`SHEPTUN_GIGAAM_RUNTIME`: **mlx** (дефолт, `gigaam-mlx` на GPU, RTF 0.02) или **onnx**
(`onnx-asr` на CPU, RTF 0.07) — выход совпадает бит в бит. Модель — `SHEPTUN_GIGAAM_MODEL`
(дефолт `gigaam-v3-e2e-ctc`), extras `.[gigaam]` / `.[gigaam-mlx]`. Пакет `gigaam` с PyPI НЕ
использовать (откатывает torch 2.9.1 → 2.5.1). **int8 стоит 3 п.п. CER** — берём fp32.
**CoreML EP падает на 13.7** (`Error computing NN outputs`).

## `_trim_silence` вредит GigaAM, но не Whisper
Обрезка тишины по краям стоит GigaAM 2 п.п. (15%→17%), паддинг не помогает; Whisper даёт 11%/14%
с ней и без. Поэтому у `_bytes_to_float_array` есть параметр `trim`, и GigaAM зовёт его с
`trim=False`. Если появится ещё один Conformer-бэкенд — начинать с `trim=False`.

## Рабочий движок — русская `v3_e2e_ctc` (боевой с 2026-09-02)
Сама расставляет пунктуацию, регистр и цифры («задержки на 2 секунды») — закрывает roadmap-пункт
«Диктовка с пунктуацией» без LLM. На живой диктовке инференс 0.09–0.3s против 0.46–0.58s у
Whisper turbo. Цена — англотермины: EPI 17% против 64% у turbo (лечится биасингом, см. ниже).
`v3_e2e_rnnt` по CER хуже (18%/20%), латиницы больше, но биасинг с ним не работает.

## Hotwords-биасинг: главный рычаг по англотерминам для CTC
`_HotwordDecoder` в `gigaam.py` — CTC beam search (`pyctcdecode`) с бустом латинских значений из
`replacements.yaml`. Даёт **CER 15%→13%, EPI 17%→34%** (200 терминов, вес 20), стоит ~+0.07s на
фразу. Ручки: `SHEPTUN_GIGAAM_HOTWORDS`, `..._LIMIT` (50→EPI 28%, 200→34%, 652→41% но 454ms),
`..._HOTWORD_WEIGHT`, `..._BEAM_WIDTH`. Работает только mlx+CTC (нужны `model.head(encoded)`).
Порядок важен: **сначала биасинг, потом сбор словаря замен** — биасинг меняет распределение
ошибок, иначе прогон `analyze-replacements` придётся делать дважды.

## EPI — метрика удержания латиницы, CER её маскирует
`_compute_epi` в `benchmark.py` + колонка в `sheptun benchmark`: 1.0 канон, 0.5 опечатка,
0.25 огрызок, 0 транслит. Whisper turbo 64%, Breeze-ASR-25 73%, GigaAM 17% — при разнице по CER
всего 4 п.п. Без EPI прогресс по терминам не виден. **Breeze-ASR-25** (файнтюн Whisper large-v2
под code-switching) при равном промпте даёт лишь +4 п.п. EPI над turbo ценой +3 п.п. CER и
двойного времени — переезд не окупается. Словарь терминов в `initial_prompt` поднимает EPI
turbo 64%→69% бесплатно.

## Смена ASR обнуляет replacements.yaml
Наши 2074 правила заточены под галлюцинации Whisper (`ритми.md`, `вапи`), а не под фонетическую
кириллицу GigaAM: прогон через пайплайн дал всего 30%→28% для мультиязычной, а выход
`v3_e2e_ctc` он прямо **ухудшает: 15%/18% → 18%/20%**. Любой переезд на другой движок = новый
прогон `analyze-replacements` с нуля. Учитывать в оценке стоимости миграции.
Связано: [[log-analyzer]], [[skill-analyzer]].

## sherpa-onnx на macOS 13.7 не работает
Их wheel собран под macOS 26.5, встроенный `libonnxruntime.dylib` требует символы CoreML новее
нашей ОС. Инференс ONNX-моделей делать на своём `onnxruntime` (1.23.2, есть CoreML EP) —
так сделан onnx-рантайм в `src/sheptun/gigaam.py`. Та же семья граблей, что и [[llm-enhancement]]
(LM Studio требует Metal 3.1).

## Фабрика рекогнайзеров одна — `create_recognizer()` в `engine.py`
Была вторая копия в `menubar.py:_init_engine` (только apple/mlx/qwen), из-за чего `gigaam` и
`parakeet` молча уезжали в `else` → CPU-Whisper: 5 секунд на фразу при живом виде «всё работает».
Новый бэкенд добавлять ТОЛЬКО в `create_recognizer`; menubar зовёт её же, а UI-прогресс загрузки
MLX живёт отдельно в `_prepare_recognizer`.

## MLX течёт по GPU-кэшу, лечится `set_cache_limit`
MLX держит отдельный GPU-буфер под КАЖДЫЙ новый размер входа и не отдаёт его: на речи разной
длины `mx.get_cache_memory()` дорос до 14 ГБ за 60 фраз при реальной потребности 1.7 ГБ, а
процесс в Activity Monitor показывал 20 ГБ. Важно: `ps`/RSS этого НЕ видит (128 МБ) — буферы
Metal считаются только в physical footprint, смотреть `vmmap -summary <pid>` (регион
IOAccelerator). Лечится `mx.set_cache_limit()` — `limit_mlx_cache()` в `recognition.py`,
зовут оба MLX-рантайма, потолок `SHEPTUN_MLX_CACHE_LIMIT_MB` (512 МБ). Побочно ускоряет
инференс: 80ms → 49ms на фразу.
