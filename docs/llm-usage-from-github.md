# Локальные LLM для ASR-коррекции — находки с GitHub

*Разведка 2026-07-05. Как OSS-проекты диктовки применяют ЛОКАЛЬНЫЕ LLM для причёсывания транскрипта, и где есть доказательства эффективности. Для каждого — репозиторий + файл. Дополняет `heuristics-from-github.md` (эвристики без LLM) и `llm-enhancement-research.md` (наш план).*

## Два главных вывода
1. **Практика вся на 1-best.** Ни одно production-приложение диктовки не использует N-best (несколько ASR-гипотез в LLM) — все чистят финальный 1-best текст. N-best→LLM живёт пока только в исследованиях (HyPoradise/GER), часто офлайн и с fine-tune.
2. **Измеренная эффективность есть только в науке.** Продакшн-апы эффективность LLM не меряют (сигнал — только звёзды/используют). Академическая линия GER даёт −26…−51% WER, НО требует LoRA-дообучения и **регрессирует zero-shot** на разговорной/акцентной речи.

Ни одного репо именно под **русскую** ASR-коррекцию локальным LLM не найдено.

---

## A. Наука — измеренный WER + готовый рецепт N-best→LLM

### A.1. HyPoradise / Hypo2Trans (H2T) ⭐ — главный рецепт N-best→LLM
- **Репо:** [Hypotheses-Paradise/Hypo2Trans](https://github.com/Hypotheses-Paradise/Hypo2Trans) (NeurIPS 2023 + ASRU 2023). Чекпоинт `GenSEC-LLM/SLT-Task1-Llama2-7b-HyPo-baseline`, датасет `PeacefulData/HyPoradise-v0`.
- **Что делает:** весь N-best список гипотез → LLM → одна исправленная транскрипция. LLaMA/LLaMA2-7B + LoRA.
- **Измеренный WER (до→после, из README):** WSJ 4.5→2.2 (−51%), ATIS 8.3→1.9 (−77%), CHiME-4 11.1→6.6 (−41%), Tedlium-3 8.5→4.6 (−46%), SwitchBoard 15.7→14.1 (−10%).
- **Критичная оговорка:** есть строки, где LLM **ухудшает** WER (CORAAL +7%, CV +4.7%, SwitchBoard +17% на некоторых вариантах). Устойчивый выигрыш — только у **LoRA-fine-tuned**; zero-shot на трудных доменах даёт регресс.
- **Ключевой аргумент за N-best:** `o_nb` (n-best oracle) заметно ниже baseline (Tedlium 3.0 vs 8.5) → правильные слова часто лежат в гипотезах 2–5, которых нет в 1-best. Файлы формата: `H2T-LoRA/`.
- **Для нас:** формат промпта/данных переиспользуем, но всё англоязычное; лучшая точность требует LoRA под русский. Начинать с 1-best безопаснее.

### A.2. RobustGER ⭐ — шумоустойчивый GER (ICLR 2024)
- **Репо:** [YUCHEN005/RobustGER](https://github.com/YUCHEN005/RobustGER). LLaMA-2 + adapter, чекпоинт `PeacefulData/RobustGER`.
- N-best + «language-space denoising» (аудио-эмбеддинг шума → языковое пространство). Цифры в paper (openreview ceATjGPTUD), не в README. Тяжёлая инфраструктура (обучение адаптера), не realtime.

Родственные (малые): `YUCHEN005/GenTranslate` (тот же приём для перевода, русский в FLEURS), `alidotdev10/Zero-Shot-GER-Code-Switched-ASR` (⭐0, zero-shot GER для code-switching через LLaMA-3.1 — концептуально близко «русский+англо», но без доказательств).

---

## B. Production-апы диктовки под Mac с ЛОКАЛЬНЫМ LLM (эффективность заявлена, не измерена)

### B.1. Arsture/whispree ⭐ — САМЫЙ близкий к нам (MLX + code-switch + guard)
- **Репо:** [Arsture/whispree](https://github.com/Arsture/whispree)
- **Движок:** Local MLX через Python-worker — `mlx-worker/mlx_llm_worker.py` (backends `mlx-lm`/`mlx-vlm`), модели Qwen3/Gemma/GLM. Ровно наш стек.
- **Промпт (1-best):** `Whispree/Services/LLM/CorrectionPrompts.swift` — *«Fix ONLY clear STT errors, MINIMUM changes, if unsure leave unchanged»* + few-shot. Есть **codeSwitchPrompt** ровно под нашу проблему: восстановление англо-терминов из мис-транслитерации (`밸리데이션→validation`, `L&M→LLM`). Аналог для русского: «эмбеддинг/реакт/юзстейт» → латиница.
- **Hallucination guard (брать обязательно):** если правка ушла дальше порога word-edit-distance от оригинала — коррекция **отбрасывается**, вставляется сырой транскрипт. Файл-описание: `docs-site/.../features/correction.md`.
- **Для нас: высокая применимость.** Прямой образец: минимальные правки + few-shot code-switch + edit-distance guard. Промпты корейско-английские — адаптировать на русский.

### B.2. giusmarci/openwhisp ⭐ — чисто локальный Ollama, надёжный рантайм
- **Репо:** [giusmarci/openwhisp](https://github.com/giusmarci/openwhisp) (Electron). Whisper (transformers.js) → Ollama → вставка.
- **Движок:** Ollama, самодостаточный HTTP-клиент `src/main/ollama.ts` (~340 строк, без SDK). Дефолт `gemma4:e4b` (`src/shared/recommendations.ts`).
- **Вызов** (`rewriteWithOllama`): `/api/chat`, `stream:false`, `keep_alive:'10m'` (тёплая модель), `temperature:0`, таймаут 90с.
- **4 уровня агрессивности** (`prompts.ts`): none/soft/medium/high; дефолт `medium` — включено и достаточно агрессивно из коробки (`defaults.ts`).
- **Анти-галлюцинация — `cleanRewriteOutput()`:** пустой ответ → вернуть сырой `rawText`; снять кавычки; регекс `metaLeadPattern` вырезает мета-преамбулы («Sure», «Here's»). Промпт `BASE_RULES`: «You are NOT a chatbot», «NO META-COMMENTARY», «NO FABRICATION», «SAME LANGUAGE».
- **Рантайм-надёжность (паттерн):** `ensureOllamaRunning()` — авто-запуск Ollama если установлена но не запущена; человекочитаемые ошибки.
- **Для нас:** образец надёжности локального рантайма + уровни агрессивности. Словаря терминов нет.

### B.3. Beingpax/VoiceInk ⭐ — самый ЗРЕЛЫЙ промпт+словарь
- **Репо:** [Beingpax/VoiceInk](https://github.com/Beingpax/VoiceInk) (см. также `voiceink-research.md`). Два локальных пути: Ollama + Local CLI.
- **Ollama:** `Services/OllamaService.swift`, `temperature:0.3`, **`think:false`** (reasoning выключен), таймаут 7с, rate-limit 1/с.
- **Local CLI:** `AIEnhancement/LocalCLIService.swift` — LLM как shell-команда (`claude -p`, `codex`, `copilot`, `pi`), промпт через env, таймаут 45с. Экзотика, для нас избыточно.
- **Эталонный промпт:** `Models/AIPrompts.swift` — самокоррекции, словесная пунктуация, списки, строгие анти-галлюцинационные правила.
- **Словарь терминов — лучший образец:** `AIEnhancementService.getSystemMessage` — секция `# Custom Vocabulary` + блок `<CUSTOM_VOCABULARY>`, формулировка «spelling authority… заменять фонетически близкие ошибки… **но не форсировать, если смысл явно другой**». Плюс контекст `<CLIPBOARD_CONTEXT>`, `<CURRENT_WINDOW_CONTEXT>`, `<CURRENTLY_SELECTED_TEXT>`.
- **Анти-reasoning:** `AIEnhancementOutputFilter.swift` вырезает `<think>/<reasoning>`; `ReasoningConfig.swift` — таблица отключения reasoning per-model (Qwen3 «none», GPT-5 «none»…).
- **Для нас:** брать промпт и словарную логику — самые проработанные из всех.

### B.4. never13254/GhostType ⭐ — каталог MLX-моделей + app-routing
- **Репо:** [never13254/GhostType](https://github.com/never13254/GhostType). Local MLX (Beta), 50+ моделей, `macos/LocalLLMCatalog.swift` (id вида `mlx-community/Qwen2.5-1.5B-Instruct-4bit`).
- Промпт `macos/PromptLibraryBuiltins.swift` — роутинг по приложению (VS Code→код-блоки, Slack→разговорный). Автор честно: локальный Qwen2.5 «works for basic cleanup, don't expect cloud-level».
- **Для нас:** готовый каталог 4bit MLX-моделей + app-aware промптинг.

### B.5. xuiltul/voice-input ⭐ — Ollama + few-shot + экранный контекст
- **Репо:** [xuiltul/voice-input](https://github.com/xuiltul/voice-input). Ollama, дефолт `gemma3:4b`, для 32GB — `qwen2.5:7b`. Whisper `large-v3-turbo`.
- Промпт `prompts/en.json` — «text formatter»: убрать филлеры, пунктуация, «fix obvious recognition errors based on context», «никогда не суммировать», few-shot (2 примера) + экранный контекст `text_context` (до 2000 симв.) для точности техтерминов. RAM ~9GB честно расписан.
- **Для нас:** рецепт Ollama + few-shot + экранный контекст под Mac.

### Не по критерию (локального LLM нет): moinulmoin/voicetypr, tover0314-w/opentypeless, kstonekuan/tambourine-voice, Murmur — «локальность» через OpenAI-совместимый base_url (Ollama как endpoint), встроенного локального рантайма нет. НО их **промпты** ценны: opentypeless `src-tauri/src/llm/prompt.rs` (мультиязык вкл. русский + словарь + анти-инъекция с санитизацией `"`/`\n`), tambourine трёхсекционный промпт `server/processors/llm.py`.

---

## Что взять для Sheptun (сводно)
1. **Начать с 1-best** (весь прод так делает, безопасно). N-best включать только если готовы LoRA-дообучать под русский — регресс zero-shot доказан таблицей HyPoradise. Но `o_nb` доказывает: N-best несёт доп. информацию → в перспективе оправдан.
2. **Промпт коррекции:** whispree `CorrectionPrompts.swift` — «минимальные правки, при сомнении не трогать» + few-shot **code-switch** (наш русский+англо).
3. **Hallucination guard (критично):** порог word-edit-distance (whispree) — правка дальше порога отбрасывается → сырой транскрипт. + `cleanRewriteOutput` (openwhisp): пустой ответ → сырой текст, вырезать мета-преамбулы/кавычки/`<think>`.
4. **Reasoning выключать явно:** консенсус (openwhisp/VoiceInk `think:false`, VoiceInk `ReasoningConfig`). Для нашего mlx-lm — прогонять без reasoning.
5. **Словарь терминов в промпт:** VoiceInk `<CUSTOM_VOCABULARY>` + opentypeless (санитизация от инъекций). «Spelling authority, не форсировать».
6. **Рантайм-надёжность:** keep_alive/тёплая модель, короткий таймаут + graceful fallback на сырой текст (openwhisp).
7. **Уровни агрессивности** none/soft/medium/high (openwhisp) — ползунок пользователю, дефолт умеренный.
8. **Экранный/app-контекст** (voice-input, GhostType) — дёшево поднимает точность техтерминов; ложится на планируемый «technical formatting».
9. **Движок:** MLX (whispree/GhostType, Qwen2.5-1.5B/3B-4bit) — мы уже подтвердили mlx-lm на 13.7. Ollama gemma3:4b/qwen2.5:7b — альтернатива.

## Ключевые файлы для копирования
- N-best формат: `Hypotheses-Paradise/Hypo2Trans/H2T-LoRA/`
- Промпт + guard: `Arsture/whispree` → `Whispree/Services/LLM/CorrectionPrompts.swift`, `docs-site/.../features/correction.md`
- Надёжный Ollama-рантайм + уровни: `giusmarci/openwhisp` → `src/main/ollama.ts`, `prompts.ts`, `defaults.ts`
- Словарь + анти-reasoning: `Beingpax/VoiceInk` → `AIEnhancementService.swift`, `AIPrompts.swift`, `AIEnhancementOutputFilter.swift`, `ReasoningConfig.swift`
- Few-shot formatter + контекст: `xuiltul/voice-input/prompts/en.json`
- MLX 4bit каталог: `never13254/GhostType/macos/LocalLLMCatalog.swift`
