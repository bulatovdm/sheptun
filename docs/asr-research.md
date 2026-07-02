# Исследование: пост-обработка ASR, лёгкие локальные модели и новые ASR для Sheptun

*Deep-research от 2 июля 2026. Приоритет: локальное, лёгкое, Apple Silicon, русский язык.*
*Метод: 5 поисковых углов → 22 источника → 104 факта → 25 верифицировано (3-голосовая adversarial-проверка, 0 опровергнуто).*

## 1. Executive summary

- **Направление 1 (пост-обработка/коррекция).** Референсная архитектура — VoiceInk: конвейер `транскрипция → детерминированные word-replacements → финальная очистка → LLM-энхансмент`, где кастомный словарь терминов подаётся в LLM тегированным блоком `<CUSTOM_VOCABULARY>` (contextual biasing на этапе пост-обработки, а не в декодере). Для русской орфо/пунктуации есть готовые открытые инструменты: **SAGE (SberDevices/ai-forever)** — семейство генеративных корректоров (орфография + пунктуация + регистр одной моделью), включая лёгкую `sage-fredt5-distilled-95m` (0.38 ГБ); и **JamSpell** — быстрый контекстный статистический спелчекер с готовой русской моделью (38 МБ).
- **Направление 2 (лёгкие локальные модели).** Для базового «причёсывания» специализированные модели точнее и легче general-LLM (Gemma ~8 ГБ): **SAGE** (0.38–1.2 ГБ) и **Silero Text Enhancement** (репунктуация + регистр для русского). Saiga лёгкой опции не даёт (минимум 7B). Данные по конкретным general-LLM 0.5–3B (Qwen/Gemma/Llama и т.д.) не верифицированы — см. Caveats.
- **Направление 3 (новые Whisper/ASR).** Для **русского** на Apple Silicon самый сильный локальный вариант — **GigaAM-v3** (Conformer 220–240M, 700K+ часов русской речи), выигрывает у Whisper-large-v3 side-by-side 70:30, MLX-порт `gigaam-mlx` (CTC ~330× RT, RNNT ~77× RT на M2 Max). Альтернатива для телефонии/коротких фраз — **T-one** (71M, стриминговый CTC, 8.63% WER на call-center против 19.39% у Whisper-large-v3). VoiceInk уже совмещает whisper.cpp и Parakeet (FluidAudio/CoreML/ANE) — мультибэкенд на macOS практичен.

---

## 2. Направление 1 — Методы пост-обработки/коррекции

### 2.1. VoiceInk: конвейер пост-обработки (confidence: high)

- **Порядок конвейера.** Word-replacements — детерминированные подстановки *после* транскрипции и форматирования абзацев, но *до* финальной очистки и LLM-энхансмента → LLM «видит» уже исправленный текст ([docs](https://tryvoiceink.com/docs/word-replacements)).
- **Матчинг замен.** Регистронезависимый; границы слова через lookaround `(?<![a-zA-Z0-9])…(?![a-zA-Z0-9])`, фолбэк на подстроку для CJK/Thai/Korean (`WordReplacementService.swift`).
- **LLM-энхансмент («AI Enhancement»).** Отдельная стадия: чистка, переписывание, пресеты Polish/Email/Chat, промпты и модель per-app через Power Mode. Вход — только текст ([power-mode](https://tryvoiceink.com/docs/power-mode)).
- **Contextual biasing на пост-обработке.** Словарь (имена, аббревиатуры, техтермины) подаётся в LLM тегом `<CUSTOM_VOCABULARY>…</CUSTOM_VOCABULARY>` как «орфографический авторитет» для замены фонетически близких ошибок (`AIEnhancementService.swift`).
- **ASR-бэкенды.** whisper.cpp + FluidAudio (Parakeet через CoreML→ANE) — несколько локальных бэкендов ([VoiceInk](https://github.com/Beingpax/VoiceInk), [FluidAudio](https://github.com/FluidInference/FluidAudio)).

**Вывод для Sheptun:** архитектура почти совпадает с текущей (replacements + фильтр галлюцинаций), но добавляет финальную очистку и LLM-энхансмент с тегированным словарём-биасингом.

### 2.2. SAGE — русская орфо/пунктуация/регистр (confidence: high)

- Открытое (MIT) семейство генеративных корректоров SberDevices/ai-forever: орфография + пунктуация + регистр одной моделью ([sage](https://github.com/ai-forever/sage), [arXiv 2308.09435](https://arxiv.org/abs/2308.09435)).
- **Модели/размеры:** `sage-fredt5-distilled-95m` — 95.6M, **0.38 ГБ** (дистилляция из FRED-T5-1.7B); `sage-fredt5-large`; `sage-m2m100-1.2B`; исторические `ruM2M100-1.2B/418M`, `FRED-T5-large-spell` ([HF](https://huggingface.co/ai-forever/sage-fredt5-distilled-95m)).
- **Качество (RUSpellRU):** `sage-m2m100-1.2B` — P 88.8 / R 71.5 / F1 79.2. Лёгкая `sage-fredt5-distilled-95m` — F1 **78.9** (орфография) / **83.6** (пунктуация) / **93.5** (регистр): почти как 1.2B при 0.38 ГБ.
- **Против конкурентов:** `ruM2M100-1.2B` обходит Yandex.Speller, HunSpell, JamSpell и OpenAI на большинстве датасетов (RUSpellRU F1 79.2 vs Yandex 69.5 / JamSpell 36.9 / HunSpell 33.0 / gpt-4 74.8), проигрывая лишь на GitHubTypoCorpusRu ([habr](https://habr.com/ru/companies/sberdevices/articles/763932/)). *Caveat:* бенчмарки 2023 г., на типографских/OCR ошибках, не ASR-специфичных.

### 2.3. JamSpell — быстрый контекстный спелчекер (confidence: high)

- Учитывает n-граммный контекст (в отличие от Norvig/Hunspell), ~5K слов/сек, C++ со SWIG-биндингами ([JamSpell](https://github.com/bakwc/JamSpell)).
- Готовая **русская модель** `ru.tar.gz` (38 МБ, 300K новостей + 300K wiki). Из коробки, но лучше дообучить.

### 2.4. Silero Text Enhancement — репунктуация и регистр (confidence: high)

- Расстановка знаков препинания и заглавных для 4 языков, включая русский ([silero-models](https://github.com/snakers4/silero-models)). *Caveat:* ~2021 г.

---

## 3. Направление 2 — Лёгкие локальные модели для очистки русского текста

Для «причёсывания» ASR-вывода **специализированные модели точнее и легче general-LLM**.

### 3.1. Специализированные корректоры вместо LLM (confidence: high)

- **SAGE** `sage-fredt5-distilled-95m` (0.38 ГБ) закрывает орфографию + пунктуацию + регистр одной моделью — ровно задача «базовой очистки», под которую рассматривалась Gemma, но в ~20× меньшем размере.
- **Silero TE** — ещё легче для узкой репунктуации/регистра.

### 3.2. Saiga не даёт лёгкой опции (confidence: high)

- Минимум в GGUF-коллекции Saiga — 7B; суб-4B нет ([saiga-gguf](https://huggingface.co/collections/IlyaGusev/saiga-gguf)). Для лёгкого «причёсывания» не подходит.

### 3.3. Общие малые LLM (0.5–3B)

- По Qwen2.5/Qwen3, Gemma 3 1B/270M, Llama 3.2 1B/3B, Phi-3.5/4-mini, SmolLM2/3, T-lite/T-pro **верифицированных данных в наборе нет** (см. Caveats). Практический вывод: general-LLM оправдан только если нужны переформулирование/адаптация под workflow (как AI Enhancement у VoiceInk); для базовой очистки — SAGE/Silero.

---

## 4. Направление 3 — Новые Whisper/ASR (июль 2026)

### 4.1. GigaAM-v3 — лучший локальный русский ASR на Apple Silicon (confidence: high)

- Conformer, 220–240M, v3 — 700K+ часов, HuBERT-CTC, варианты CTC и RNNT ([GigaAM](https://github.com/salute-developers/GigaAM), [HF](https://huggingface.co/ai-sage/GigaAM-v3), [arXiv 2506.01192](https://arxiv.org/abs/2506.01192)).
- **Качество:** e2e CTC/RNNT выигрывают у Whisper-large-v3 side-by-side 70:30 (LLM-as-a-Judge). *Caveat:* вендорская метрика.
- **Apple Silicon:** MLX-порт `gigaam-mlx` (без PyTorch): CTC ~330× RT, RNNT ~77× RT на M2 Max ([gigaam-mlx](https://github.com/aystream/gigaam-mlx)). *Caveat:* пик для коротких клипов; на длинном аудио ~50×/42× RT; порт сторонний.
- **Доп. бенчмарк (не верифицирован):** habr сообщает 3.3% WER на CPU против 7.9% у Whisper-large-v3-turbo, в 2.4× быстрее ([habr 1002260](https://habr.com/ru/articles/1002260/)).

### 4.2. T-one — стриминговый русский ASR (confidence: high)

- 71M, Conformer+CTC, стриминг, ~22× меньше Whisper-large-v3 ([T-one](https://github.com/voicekit-team/T-one), [HF](https://huggingface.co/t-tech/T-one)).
- Call-center: **8.63% WER vs 19.39%** у Whisper-large-v3. *Caveat:* доменный бенчмарк (на CommonVoice общей речи Whisper выигрывает); вендорские цифры.

### 4.3. Мультибэкенд whisper.cpp + Parakeet/CoreML/ANE (confidence: high)

- VoiceInk совмещает whisper.cpp и Parakeet через FluidAudio (CoreML→ANE) — несколько локальных бэкендов на macOS жизнеспособны.
- Бенчмарк-репо: [mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest) (8 бэкендов на одном аудио; для `large`: fluidaudio-coreml 0.19s > parakeet-mlx 0.50s > …), [onnx-asr](https://github.com/istupakov/onnx-asr).

---

## 5. Рекомендации для Sheptun

| # | Что применить | Напр. | Размер / RAM | Сложность | Ссылки |
|---|---|---|---|---|---|
| R1 | **SAGE `sage-fredt5-distilled-95m`** — пунктуация+регистр+орфография после word-replacements | 1, 2 | 0.38 ГБ | Низкая-средняя (HF/CTranslate2, MLX/torch-mps) | [HF](https://huggingface.co/ai-forever/sage-fredt5-distilled-95m), [sage](https://github.com/ai-forever/sage) |
| R2 | **JamSpell (ru)** — быстрый контекстный спелчекер в горячем пути (латентность критична) | 1 | 38 МБ | Низкая (C++/Python) | [JamSpell](https://github.com/bakwc/JamSpell) |
| R3 | **Паттерн VoiceInk**: `replacements → cleanup → (опц.) LLM-энхансмент`, словарь в LLM тегом `<CUSTOM_VOCABULARY>` | 1, 2 | зависит от LLM | Средняя | [docs](https://tryvoiceink.com/docs/word-replacements), [power-mode](https://tryvoiceink.com/docs/power-mode) |
| R4 | **GigaAM-v3 MLX** — оценить как основной русский ASR-бэкенд вместо/рядом с Whisper | 3 | ~220M (≈0.5–1 ГБ) | Средняя (MLX-рантайм, замена recognizer) | [gigaam-mlx](https://github.com/aystream/gigaam-mlx), [GigaAM](https://github.com/salute-developers/GigaAM) |
| R5 | **T-one** — альтернатива для стриминга/коротких команд | 3 | 71M | Средняя | [T-one](https://github.com/voicekit-team/T-one) |
| R6 | **Silero TE** — минимальный fallback репунктуации, если SAGE тяжёл | 1, 2 | малый | Низкая | [silero-models](https://github.com/snakers4/silero-models) |

**Приоритет внедрения:** R2 (дёшево, сразу) → R1 (главный выигрыш по качеству текста) → R4 (главный выигрыш по WER для русского) → R3 (опц., если нужен LLM-энхансмент) → R5/R6.

---

## 6. Caveats

- **Общие малые LLM (0.5–3B):** по Qwen/Gemma 3/Llama 3.2/Phi/SmolLM/T-lite/T-pro **верифицированных данных нет** — направление требует дорасследования.
- **Часть ASR:** distil-whisper, large-v3-turbo, Parakeet/Canary, Moonshine, Kyutai STT, Voxtral, Vosk, pisets — прямых верифицированных claims нет (Parakeet — лишь как бэкенд VoiceInk). Сводных WER-таблиц для русского по всем моделям в наборе нет.
- **Вендорские метрики:** GigaAM 70:30 — LLM-as-a-Judge, не аудированный WER; T-one/GigaAM цифры самозаявлены.
- **Доменность:** T-one 8.63% — только call-center; на общей речи Whisper выигрывает. Не экстраполировать на терминальные команды без своего теста.
- **SAGE и ASR:** обучен на типографских/OCR ошибках, не ASR-специфичных; применимость к Whisper — обоснованный вывод, не бенчмарк. RUSpellRU — 2023 г.
- **Скорость GigaAM MLX:** ~330×/77× — пик для коротких клипов; длинное аудио ~50×/42×. `gigaam-mlx` — сторонний порт.
- **Silero TE:** ~2021 г. **VoiceInk docs:** часть URL (`power-mode`) иногда 404, содержимое подтверждено исходниками.

## 7. Открытые вопросы

1. Как SAGE/JamSpell ведут себя именно на ASR-ошибках Whisper для русского (не типографских), на коротких терминальных командах с техтерминами/англицизмами? Нужен мини-бенчмарк на логах Sheptun.
2. GigaAM-v3 vs Whisper на доменной лексике Sheptun (команды, имена файлов, snake_case, ru/en) — как выигрыш 70:30 транслируется на управление терминалом?
3. Нужен ли вообще general-LLM-энхансмент, или связки SAGE+JamSpell (0.4 ГБ, детерминированнее и быстрее) достаточно при жёстких требованиях к латентности?
4. Латентность пайплайна end-to-end на Apple Silicon: GigaAM(MLX) → replacements → SAGE/JamSpell — укладывается ли в интерактивный бюджет, какой квант/рантайм оптимален для 95M-корректора (MLX vs CTranslate2 vs llama.cpp)?
