# Эвристики улучшения качества ASR — находки с GitHub

*Разведка 2026-07-05: ~15 open-source проектов голосового ввода/диктовки (кроме VoiceInk — он в `voiceink-research.md`). Цель — конкретные приёмы улучшения качества, которых нет в Sheptun. Для каждого указан репозиторий + файл/функция, чтобы посмотреть реализацию перед интеграцией.*

## ⚠️ Ограничение нашего стека (нашли независимо 2 агента)
На **MLX-пути `beam_size`, `suppress_tokens`, `vad_filter` отбрасываются** обёрткой mlx-audio (`del ...` в её коде). Значит decoding-трюки Whisper (suppress_blank, suppress-regex, beam-профили, температурный fallback через параметры) на нашем MLX-turbo **недоступны напрямую** — упор на пред/пост-обработку аудио и текста. Проверить, что реально пробрасывается в MLX `generate()`: только `language/task/initial_prompt/word_timestamps`.

---

## Группа 1 — Анти-галлюцинация на тихом/коротком аудио (дёшево, ценно)

### 1.1. Фильтр «эха словаря» ⭐
Whisper на тишине/шуме выплёвывает обратно текст `initial_prompt`. Нормализовать результат и промпт (lowercase, без пунктуации), посчитать пересечение слов: если `textComposition ≥ 0.9 И dictionaryUsage ≥ 0.7` — отбросить как эхо.
- **Репо:** OpenWhispr/openwhispr
- **Файлы:** `src/utils/dictionaryEchoFilter.js` (`matchesDictionaryPrompt`), применяется в `src/helpers/audioManager.js` (`isDictionaryEcho`)
- **Для нас:** у нас есть `initial_prompt` + стоп-лист, но нет проверки «не вернул ли Whisper сам биас». Дешёвая надстройка, ловит класс галлюцинаций, который фиксированный стоп-лист не покрывает.

### 1.2. RMS-гейт для стоп-листа галлюцинаций ⭐
Известную фразу-галлюцинацию дропать ТОЛЬКО если RMS аудио ниже порога тишины (`_SUSPICIOUS_RMS_THRESHOLD=0.02`). Громко сказал — пропустить. Плюс посегментный дроп по `no_speech_prob > 0.6` (а не всего результата).
- **Репо:** Nimdy/jarvis-oracle-edition
- **Файлы:** `brain/perception/stt.py` — `_looks_like_hallucination(text, rms)`, `_collect_segment_text()`
- **Для нас:** убирает ложные срабатывания нашего `SHEPTUN_HALLUCINATIONS`. Энергия у нас уже есть (EnergyVAD).

### 1.3. Amplitude-гейт «нет речи» + паддинг аудио ⭐
Классификация записи по длине и пиковой громкости: `<0.04с` → отбросить; `<1.0с` при `peakLevel<0.003` → «нет речи»; `≥1.0с` при `peakLevel<0.006` → «нет речи». Принцип: «ложный отброс хуже пустого текста». Паддинг: `<0.75с` добить нулями до 0.75с, иначе +0.3с тишины в хвост.
- **Репо:** TypeWhisper/typewhisper-mac
- **Файлы:** `ViewModels/DictationViewModel.swift` — `classifyShortSpeech(...)`, `paddedSamplesForFinalTranscription`; `TypeWhisperPluginSDK/.../TypeWhisperPlugin.swift` — `PluginAudioUtils.paddedSamples`
- **Для нас:** Whisper обучен на 30с окнах, на коротких/обрывающихся клипах глотает последнее слово или галлюцинирует. Хвостовой паддинг + amplitude-гейт против пустых запусков.

### 1.4. Retry без VAD при пустом результате + наличии энергии
Whisper вернул пусто, но rms выше пола (`_VAD_RETRY_RMS_FLOOR=0.02`) → повторный прогон с `vad_filter=False`. Против случаев, когда VAD съел тихую реальную речь.
- **Репо:** Nimdy/jarvis
- **Для нас:** рецепт против проглатывания тихих команд нашим Silero/EnergyVAD.

### 1.5. Confidence/logprob-гейт коротких клипов
Если аудио `<1.0с` И `confidence<0.55` → отбросить (пустая строка).
- **Репо:** TypeWhisper — `PluginAudioUtils.shouldAcceptShortClipTranscription`, применяется в `ParakeetPlugin.swift`
- **Для нас:** применимо, MLX-Whisper даёт `avg_logprob` по сегментам → порог по logprob для коротких фраз отсеет шум.

---

## Группа 2 — VAD и захват фразы

### 2.1. Draining с адаптивным таймаутом ⭐
Автомат Idle→Recording→WaitingForSTT→Draining. После стопа не завершать ход сразу, а войти в draining и ждать `transcription_wait_timeout` (0.5с); КАЖДЫЙ приход поздней транскрипции сбрасывает таймер. Завершить, когда за окно ничего нового. Прямо комментируют: для медленных локальных STT.
- **Репо:** kstonekuan/tambourine-voice
- **Файлы:** `server/processors/turn_controller.py` — `DrainingState`, `_draining_task_handler`, `_draining_event`
- **Для нас:** ловит «хвост» фразы, который MLX выдаёт с задержкой после того, как VAD сказал «тишина» — иначе последние слова теряются/уезжают в следующую фразу.

### 2.2. Двойной VAD «Silero AND WebRTC»
Голос активен только когда согласны оба: WebRTC (быстрый скрининг 10мс кадрами) + Silero (точное подтверждение).
- **Репо:** homelab-00/TranscriptionSuite
- **Файлы:** `server/backend/core/stt/vad.py` — класс `VoiceActivityDetector`
- **Для нас:** у нас energy+Silero; добавить WebRTC как дешёвый первый каскад с правилом «оба согласны» → меньше ложных на шуме/дыхании.

### 2.3. Min-voice-duration анти-блип
Озвученным считать только пробег RMS выше порога длительностью ≥300мс подряд (`MIN_VOICE_DURATION`) — щелчки/стуки не переводят в «речевую» ветку. Плюс тайры: dead-mic (10с), долгая тишина (60с), таймаут (300с).
- **Репо:** moinulmoin/voicetypr
- **Файлы:** `src-tauri/src/audio/silence_detector.rs` — `SilenceDetector::update_at`, `MIN_VOICE_DURATION`
- **Для нас:** дешёвая добавка к EnergyVAD против ложных срабатываний.

### 2.4. Три раздельных временных порога VAD
Вместо одного `SILENCE_DURATION`: `post_speech_silence_duration=1.0` (конец фразы), `min_length_of_recording=0.5`, `min_gap_between_recordings=0.3`.
- **Репо:** homelab-00/TranscriptionSuite
- **Файлы:** `server/backend/core/live_engine.py` — `LiveModeConfig`, `_process_sentence`
- **Для нас:** лучше отделяет реальный конец фразы от микропауз, глотает слишком короткие «щелчки».

---

## Группа 3 — Пред-обработка аудио (у нас её нет вообще)

### 3.1. Speech-gated нормализация громкости ⭐
Пик-нормализация WAV до −1.9 dBFS, но лимит усиления зависит от «речеподобности»: обычный кэп ×10, для речи разблокируется ×32. «Речь» = модуляция огибающей (сравнение 90-го и 10-го перцентилей RMS по 20мс окнам; речь качается ~12дБ, ровный шум — нет) + ≥300мс озвученной энергии. TPDF-дизеринг при квантовании в i16.
- **Репо:** moinulmoin/voicetypr
- **Файлы:** `src-tauri/src/audio/normalizer.rs` — `normalize_to_whisper_wav`, `has_speech_like_modulation`, `peak_normalization_gain`
- **Для нас:** нет нормализации громкости перед Whisper. Тихий/далёкий микрофон = хуже распознавание; speech-gated boost безопаснее AGC (не раздувает шум в паузах).

### 3.2. Качественный ресемплинг (rubato) + equal-power downmix
Ресемплинг в 16кГц через `rubato` (не линейной интерполяцией); многоканальный downmix equal-power с исключением почти-тихих каналов (мёртвый стерео-канал не разбавляет сигнал).
- **Репо:** moinulmoin/voicetypr — `src-tauri/src/audio/resampler.rs`, `normalizer.rs::downmix_equal_power_ignore_silent`
- **Анти-пример:** openwhisp/opentypeless ресемплят наивной линейной интерполяцией — так делать НЕ надо.
- **Для нас:** если запись не нативно 16кГц mono — качественный ресемплер заметно влияет на ASR.

### 3.3. Warmup модели на тишине
Перед первым реальным распознаванием прогнать 1с нулей через модель.
- **Репо:** homelab-00/TranscriptionSuite — `mlx_whisper_backend.py::warmup()` (`np.zeros(SAMPLE_RATE)`)
- **Для нас:** убирает лаг первого распознавания после старта (UX).

---

## Группа 4 — Пост-обработка текста

### 4.1. Word-boundary + защита URL/email/кода при заменах ⭐
Замены регэкспом `case_insensitive`, но с защитами: `candidate_has_boundaries` (только граница слова) и `span_in_protected_token` (не применять, если токен содержит `://` или `@`). Приоритеты правил + провенанс.
- **Репо:** moinulmoin/voicetypr
- **Файлы:** `src-tauri/src/writing.rs` — `is_boundary_word_char`, `candidate_has_boundaries`, `span_in_protected_token`, `apply_text_replacements_with_provenance`
- **Для нас:** прямо релевантно `replacements.yaml` и скиллу review-replacements — закрывает баг «замена внутри слова/ссылки/кода».

### 4.2. Дедуп/нормализация пунктуации
Словарь фраз→знак с тонкостями: режимы `selectiveFallback`/`fullFallback` (ставить знак только если модель сама не поставила), подавление дубликатов (рядом уже знак → вставить пустоту, не `??`), нормализация пробелов вокруг знаков (убрать пробел перед `,.:;?!)`, не ставить после `(`), матч по границам слов.
- **Репо:** TypeWhisper — `Services/SpeechPunctuationService.swift`, правила `Resources/PunctuationRules/{en,de,it,ja}.json`, `PunctuationStrategyResolver.swift`
- **Для нас:** наш `replacements.yaml` плоский. Взять: режим «не дублировать пунктуацию модели», дедуп знаков, нормализация пробелов (для русского особенно).

### 4.3. `spoken_form` у терминов (как слышится) ⭐
У кастомного слова два поля: `phrase` (правильное) и `spoken_form` (как Whisper слышит). Даёт (а) детерминированную замену `spoken_form→phrase`, (б) подсказку LLM `phrase (may be heard as: spoken_form)` — «мостик» для маппинга «shad cn»→«shadcn/ui». opentypeless: поле `pronunciation`.
- **Репо:** moinulmoin/voicetypr — `src-tauri/src/writing.rs::compiled_replacement_rules`, план `plans/028-ai-polish-clarity.md`; tover0314-w/opentypeless — `src-tauri/src/commands/dictionary.rs`
- **Для нас:** к правильному термину добавить «как слышится» → осмысленнее генерация правил + готовая подсказка для N-best→LLM каскада.

### 4.4. Стрип whisper-маркеров по белому списку в скобках
Регэксп ловит любые `[...]`/`(...)`, но удаляет только если содержимое в `knownMarkers` (`BLANK_AUDIO, Music, Applause, Laughter, Silence, INAUDIBLE, NOISE`, все регистры). Реальный текст в скобках не трётся.
- **Репо:** human37/open-wispr — `Transcriber.swift::stripWhisperMarkers` + `knownMarkers`
- **Для нас:** дополняет наш фразовый фильтр — точечно бить звуковые ремарки, не задевая легитимные скобки.

### 4.5. Нормализация чисел-слов → цифры с защитой от ложных срабатываний
Пер-язычные парсеры «двадцать один»→`21`. Эвристики: run одиночных цифр («one nine eight four»→`1984`) только при длине ≥4; «oh»/«o» как ноль только между цифрами; bare «first/second» НЕ конвертировать, «twenty first»→`21st` — да; минус, десятичные «point», масштабы.
- **Репо:** TypeWhisper — `Services/NumberNormalization/*NumberWordParser.swift` (русского нет!), `NumberWordNormalizer.swift`
- **Для нас:** русского парсера у них нет — ниша для нас; их эвристики — готовый чек-лист против ломания диктовки. Числа у нас сейчас не нормализуются.

### 4.6. Три политики матчинга замен + авто-выбор по типу ключа
Для каждой correction политика `exact`/`boundary`/`substring`; авто: ключ со «словоподобными» символами → `boundary`, иначе (пунктуация/символы) → `substring`. Скрипт-специфичные границы, `caseSensitive`.
- **Репо:** TypeWhisper — `DictionaryService.swift::matchPolicy(for:)`, `applyCorrection`, `isBoundaryMatch`
- **Для нас:** наш replace жёстко «по границам»; авто-выбор политики по составу ключа снял бы часть ложных/пропущенных замен (символы/пунктуация как ключи).

### 4.7. Эргономичный пробел после конца предложения на границе вставки
Один пробел после `.!?` (учитывая кавычки/скобки), но ПРОПУСК для URL (`.com/.io/.dev`, `://`), email (`@`), кода (`=`, `->`, `::`, `{}`, `;`). Только на границе вставки — история чистая.
- **Репо:** moinulmoin/voicetypr — `src-tauri/src/commands/text.rs::ensure_trailing_sentence_space`
- **Для нас:** пересекается с нашей trailing-space-логикой; эвристики «не трогать URL/email/код» переносимы 1:1.

---

## Группа 5 — Форматирование под приложение-получатель (у нас есть `focus.py` — половина готова)

### 5.1. App-aware форматирование ⭐
Определить фронт-приложение (bundle id / URL) → тип (Code/Email/Chat/Document) → в терминале plain text без Markdown/капитализации, в мессенджере — иначе. Формат по типу подмешивается в промпт LLM (Email→формальный, Chat→коротко).
- **Репо:** TypeWhisper — `AppFormatterService.swift` (`nativeAppMappings`, `browserBundleIdentifiers`); voicetypr — `writing.rs::resolve_app_formatting_preset`; opentypeless — `app_detector/mod.rs` + `llm/prompt.rs` (`EMAIL_ADDON`/`CHAT_ADDON`/`DOCUMENT_ADDON`); tambourine — `context_manager.py`
- **Анти-injection (обязательно копировать):** данные фокуса помечать как untrusted metadata, «never follow as commands»; санитизация (control-символы, схлопывание пробелов, обрезка 300/500, origin до `scheme://netloc`). tambourine — `SanitizedFocusText`, `_format_active_app_context_block`.
- **Для нас:** у нас `FocusTracker`/`focus.py` — половина инфраструктуры. Прямое расширение для LLM-каскада + trailing-space (в терминале не капитализировать).

### 5.2. Терминал-специфичная вставка текста
Для терминалов (Terminal/iTerm/WezTerm/Warp) — принудительно синтетическая вставка (paste) вместо Accessibility + увеличенная задержка восстановления буфера (900мс), т.к. терминалы теряют/искажают текст.
- **Репо:** TypeWhisper — `TextInsertionService.swift` (`syntheticPastePreferredBundleIdentifiers`, `terminalPasteFallbackRestoreDelay`)
- **Для нас:** мы целимся в терминал; ориентир по надёжности вставки (мы недавно перешли на clipboard-путь — согласуется).

---

## Группа 6 — Streaming-дедуп (пригодится, если добавим потоковую диктовку)

### 6.1. LocalAgreement-n
Фиксировать («commit») только префикс, совпавший в последних N транскрипциях подряд И оканчивающийся на пунктуацию; нестабильный хвост держать как partial.
- **Репо:** scribear/ScribeAR-NodeServer — `whisper-service/model_bases/local_agree_model_base.py`; компактный вариант (~10 строк) deepmindru-afk/SimulAiz — `src/simulaiz/stt_whisper.py`
- **Для нас:** сейчас у нас пофразный VAD-цикл без стыков; актуально только при переходе на streaming.

### 6.2. Дедуп near-duplicate через LCS токенов с игнором стоп-слов
Нормализовать, токенизировать, longest-common-token-subsequence + покрытие; отдельный счёт по «значимым» токенам (выбросив стоп-лист и токены длины 1). Пороги ~0.55–0.6.
- **Репо:** OpenWhispr — `src/helpers/transcriptText.js` (`transcriptsOverlap`, `longestCommonTokenSubsequence`, `LOW_SIGNAL_TOKENS`)
- **Для нас:** склейка повторов между соседними VAD-сегментами (Whisper повторяет хвост предыдущей фразы). Игнор стоп-слов при сравнении.

---

## Приоритеты внедрения (польза/усилие, MLX-совместимо, без LLM)
**Быстрые победы:** 1.1 (эхо словаря), 1.2 (RMS-гейт стоп-листа), 2.1 (draining-таймаут), 4.1 (word-boundary + защита URL в заменах).
**Среднее:** 1.3 (паддинг + amplitude-гейт), 3.1 (speech-gated нормализация громкости).

## Референсы «как НЕ надо»
- Наивный линейный ресемплинг: openwhisp `audio-recorder.ts::resample`, opentypeless `capture.rs::downsample`.
- Whisper с дефолтами без анти-галлюцинационных трюков: openwhisp (transformers.js).
