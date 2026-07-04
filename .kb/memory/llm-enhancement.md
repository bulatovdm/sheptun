# LLM Enhancement (локальное причёсывание транскрипта)

## Исследование живёт в docs/llm-enhancement-research.md
Опциональный пост-процесс поверх ASR: прогон транскрипта через локальную LLM (пунктуация, капитализация, грамматика, паразиты, устные самокоррекции) — аналог «AI Enhancement» у VoiceInk. **Не реализовано**, только исследовано (2026-07-04). Подробности, тесты и промпты — в `docs/llm-enhancement-research.md`. В roadmap — пункт в «Долгосрочно».

## Ключевые грабли (чтобы не переоткрывать)
- **reasoning_effort:"none"** в теле запроса полностью гасит «раздумья» thinking-моделей (Gemma QAT: 18-51с → ~1с). `chat_template_kwargs:{enable_thinking:false}` только снижает. Без reasoning падает качество — лечится few-shot правилами с примерами в промпте.
- **LLM галлюцинирует доменные термины без словаря** (`гема`→`схема`!). Термины надо давать ЯВНО в промпте (компактный список 5-10, НЕ весь `replacements.yaml`: 2074 правила ~137k токенов не влезут, а выжимка 639 терминов перегружает и ломает причёсывание). Архитектура: `replacements.yaml` детерминированным слоем ДО LLM.
- **LM Studio несовместим с macOS 13.7**: движки (llama.cpp Metal v2.23.1, MLX v1.9.1) собраны под macOS 14+/**Metal 3.1**. Симптомы: GGUF Qwen — `GGML_ASSERT(buf_dst)` в `ggml_metal_cpy_tensor_async`; MLX — `invalid value 'metal3.1' in '-std=metal3.1'`. Обход: свой `mlx-lm` в venv (как наш MLX-Whisper, собран под систему) или откат рантайма LM Studio на старый (Metal 3.0).

## Наш ASR: Whisper через MLX на GPU (НЕ на CPU)
`SHEPTUN_RECOGNIZER=mlx`, `SHEPTUN_MODEL=turbo` → `MLXWhisperRecognizer` → `mlx-community/whisper-large-v3-turbo`. Работает через MLX на GPU M2 Max. Важно: PyTorch-MPS на macOS 13.7 недоступен (`torch.backends.mps.is_available()==False`), НО MLX — отдельный стек и работает. Не путать «MPS недоступен» с «GPU недоступен»: у нас M2 Max, 30-ядерный GPU, Metal 3 — GPU есть, Whisper на нём.
Связано: `src/sheptun/recognition.py` (`MLXWhisperRecognizer`, `MLX_MODELS`), `src/sheptun/engine.py:316`.
