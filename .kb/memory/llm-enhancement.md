# LLM Enhancement (локальное причёсывание транскрипта)

## Исследование живёт в docs/llm-enhancement-research.md
Опциональный пост-процесс поверх ASR: прогон транскрипта через локальную LLM (пунктуация, капитализация, грамматика, паразиты, устные самокоррекции) — аналог «AI Enhancement» у VoiceInk. **Не реализовано**, только исследовано (2026-07-04). Подробности, тесты и промпты — в `docs/llm-enhancement-research.md`. В roadmap — пункт в «Долгосрочно».

## ПОДТВЕРЖДЕНО (2026-07-05): свой mlx-lm в venv работает на 13.7
Обходной путь №1 проверен и работает. `pip install mlx-lm` (0.30.2, тянет тот же `mlx 0.29.3`, что и наш MLX-Whisper — собран под систему). Прямой Python API `from mlx_lm import load, generate` — **НЕТ ошибки metal3.1**, генерация на GPU: загрузка Qwen2.5-7B-Instruct-4bit ~3с, инференс фраз 0.6–1.7с. То есть enhancement можно делать своим MLX, минуя LM Studio целиком. Побочка установки: `mlx-lm` обновил transformers→5.0rc1, huggingface-hub→1.22, tokenizers→0.22 — наш Whisper/пакет после этого импортируются (проверено), но если что-то в HF-путях сломается — смотреть сюда.
**Качество Qwen2.5-7B-4bit — слабовато:** термины из промпта применяет (виспле→Whisper, гема→Gemma, voice-ing→VoiceInk ✅), но теряет самокоррекцию «нет стоп», не делает словесную пунктуацию, и ГАЛЛЮЦИНИРУЕТ (на «IE enhancement» дописал целый выдуманный абзац про «Integrated Environment»). Для боевого качества нужна модель крупнее (Qwen2.5-14B-4bit влезет в 32GB, или Gemma-2-9b MLX) + few-shot промпт с правилом «ничего не дописывай».
Модели MLX качаются в `~/.cache/huggingface/hub/` (общий кэш с Whisper). Тестовые Qwen удалены после проверки, чтобы не занимали место.

## Ключевые грабли (чтобы не переоткрывать)
- **reasoning_effort:"none"** в теле запроса полностью гасит «раздумья» thinking-моделей (Gemma QAT: 18-51с → ~1с). `chat_template_kwargs:{enable_thinking:false}` только снижает. Без reasoning падает качество — лечится few-shot правилами с примерами в промпте.
- **LLM галлюцинирует доменные термины без словаря** (`гема`→`схема`!). Термины надо давать ЯВНО в промпте (компактный список 5-10, НЕ весь `replacements.yaml`: 2074 правила ~137k токенов не влезут, а выжимка 639 терминов перегружает и ломает причёсывание). Архитектура: `replacements.yaml` детерминированным слоем ДО LLM.
- **LM Studio несовместим с macOS 13.7**: движки (llama.cpp Metal v2.23.1, MLX v1.9.1) собраны под macOS 14+/**Metal 3.1**. Симптомы: GGUF Qwen — `GGML_ASSERT(buf_dst)` в `ggml_metal_cpy_tensor_async`; MLX — `invalid value 'metal3.1' in '-std=metal3.1'`. Обход: свой `mlx-lm` в venv (как наш MLX-Whisper, собран под систему) или откат рантайма LM Studio на старый (Metal 3.0).

## Наш ASR: Whisper через MLX на GPU (НЕ на CPU)
`SHEPTUN_RECOGNIZER=mlx`, `SHEPTUN_MODEL=turbo` → `MLXWhisperRecognizer` → `mlx-community/whisper-large-v3-turbo`. Работает через MLX на GPU M2 Max. Важно: PyTorch-MPS на macOS 13.7 недоступен (`torch.backends.mps.is_available()==False`), НО MLX — отдельный стек и работает. Не путать «MPS недоступен» с «GPU недоступен»: у нас M2 Max, 30-ядерный GPU, Metal 3 — GPU есть, Whisper на нём.
Связано: `src/sheptun/recognition.py` (`MLXWhisperRecognizer`, `MLX_MODELS`), `src/sheptun/engine.py:316`.
