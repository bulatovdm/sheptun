# VoiceInk — исследование и идеи для интеграции

**Источник:** [github.com/Beingpax/VoiceInk](https://github.com/Beingpax/VoiceInk)
**Стек:** Swift, macOS 14.4+, whisper.cpp, SwiftUI
**Лицензия:** GPLv3 (3.7k stars, 114 релизов)

VoiceInk — нативное macOS-приложение для voice-to-text с локальным Whisper.
Ниже — конкретные фичи и подходы, которые стоит реализовать в Sheptun.

---

## Высокий приоритет

### 1. Языкоспецифичный initial_prompt для русского

**Что делает VoiceInk:** передаёт в Whisper `initial_prompt` с текстом на целевом языке. Для русского — `"Здравствуйте, как ваши дела? Приятно познакомиться."`. Это подсказывает модели язык и стиль, значительно улучшая качество распознавания.

**Куда в Sheptun:** `recognition.py` — параметр `initial_prompt` при вызове `model.transcribe()`.

**Что писать в prompt:** русскоязычный текст с терминологией, характерной для Sheptun — команды терминала, голосовые команды. Например:
```
"Здравствуйте. Открой терминал. Скопируй текст. Вставь. Отмени. Клод, напиши функцию."
```

**Сложность:** минимальная (5 строк кода)

### 2. Улучшенный фильтр галлюцинаций

**Что делает VoiceInk (`TranscriptionOutputFilter`):**
- Regex для удаления текста в `[квадратных скобках]` — Whisper пишет `[музыка]`, `[тишина]`
- Regex для `(круглых скобок)` — `(смех)`, `(аплодисменты)`
- Regex для `{фигурных скобок}`
- Удаление XML-тегов типа `<|en|>`, `<|0.00|>`
- Настраиваемый список filler words (uh, um, hmm)
- `FillerWordManager` — пользователь может добавлять/удалять слова-паразиты

**Куда в Sheptun:** `recognition.py` — расширить `SHEPTUN_HALLUCINATIONS`:
```python
# Паттерны-regex для фильтрации
HALLUCINATION_PATTERNS = [
    r'\[.*?\]',          # [музыка], [тишина]
    r'\(.*?\)',          # (смех), (аплодисменты)
    r'\{.*?\}',          # {неразборчиво}
    r'<\|.*?\|>',       # <|en|>, <|0.00|>
]
```

**Сложность:** низкая

### 3. WordReplacement — замена слов после транскрипции

**Что делает VoiceInk (`WordReplacementService`):**
- Case-insensitive замена слов/фраз в тексте после транскрипции
- Поддержка нескольких вариантов написания через запятую (CSV)
- Хранение в SwiftData (у нас — в YAML-конфиге)

**Куда в Sheptun:** секция `replacements` в `sheptun.yaml`:
```yaml
replacements:
  клот: клод
  питон: python
  гитхаб: GitHub
  бэш: bash
  вскод: VS Code
```

Применяется в `recognition.py` или `commands.py` после транскрипции, до парсинга команд.

**Сложность:** низкая

### 4. Dual-mode PTT / Hands-Free по одной клавише

**Что делает VoiceInk (`HotkeyManager`):**
- Одна горячая клавиша, два режима:
  - **Короткое нажатие** (< 0.5 сек) — toggle hands-free (нажал = запись, нажал снова = стоп)
  - **Длинное нажатие** (>= 0.5 сек) — push-to-talk (удерживаешь = запись, отпустил = стоп)
- Определение через замер `pressDuration = time() - startTime`

**Куда в Sheptun:** `hotkeys.py` — заменить текущую логику PTT на dual-mode.

**Сложность:** средняя (нужно аккуратно с таймингами)

---

## Средний приоритет

### 5. Audio metering — визуализация уровня микрофона

**Что делает VoiceInk (`Recorder.swift`):**
- Замер уровня звука каждые 17мс
- Нормализация dB → диапазон 0..1
- EMA-сглаживание: `smoothed = smoothed * 0.6 + current * 0.4`

**Куда в Sheptun:** `status.py` — Rich progress bar с уровнем микрофона. Помогает пользователю понять, что микрофон работает и слышит голос.

**Сложность:** средняя

### 6. Детекция "нет звука"

**Что делает VoiceInk:** если через 5 секунд записи не обнаружен звук — уведомление "No Audio Detected". Через 12 секунд — повторное предупреждение.

**Куда в Sheptun:** `audio.py` — предупреждение "Микрофон не ловит звук. Проверь разрешения или выбранное устройство."

**Сложность:** низкая

### 7. Trigger words — ключевые слова для переключения режимов

**Что делает VoiceInk (`PromptDetectionService`):**
- Определённые слова в начале/конце фразы автоматически активируют AI-промпт
- Trigger words задаются пользователем для каждого промпта

**Куда в Sheptun:** `commands.py` — слово-префикс определяет режим:
- "запиши ..." → режим диктовки (текст как есть)
- "выполни ..." → режим команд
- "клод ..." → отправка в Claude Code

**Сложность:** средняя

### 8. Model prewarm — прогрев модели при запуске

**Что делает VoiceInk (`ModelPrewarmService`):**
- При запуске приложения и после пробуждения из сна прогоняет короткий аудио-фрагмент через модель
- Это компилирует граф для ANE (Apple Neural Engine) и кеширует его
- Первая реальная транскрипция работает значительно быстрее

**Куда в Sheptun:** `engine.py` — при старте выполнить dummy-транскрипцию пустого аудио.

**Сложность:** низкая

### 9. Mute системных звуков во время записи

**Что делает VoiceInk (`Recorder.swift`):**
- Приглушает системные звуки (уведомления) во время записи
- Ставит на паузу медиа-воспроизведение
- Восстанавливает после окончания записи

**Куда в Sheptun:** `audio.py` — через `osascript` управлять системной громкостью alert volume.

**Сложность:** низкая

---

## Низкий приоритет (большой объём работы)

### 10. Power Mode — контекстные профили по приложениям

**Что делает VoiceInk:**
- Автоматическое переключение настроек (модель, язык, AI-промпт, горячая клавиша) в зависимости от активного приложения
- Определение приложения через `NSWorkspace.frontmostApplication`
- Определение URL из 11 браузеров через AppleScript
- Сохранение/восстановление состояния сессии

**Куда в Sheptun:** расширить `FocusTracker` из `focus.py` до системы профилей. Конфигурация в `sheptun.yaml`:
```yaml
profiles:
  terminal:
    apps: [com.apple.Terminal, com.googlecode.iterm2]
    commands: terminal-commands.yaml
    model: medium
  browser:
    apps: [com.apple.Safari, com.google.Chrome]
    commands: browser-commands.yaml
```

**Сложность:** высокая

### 11. Абстракция TranscriptionModel — поддержка облачных провайдеров

**Что делает VoiceInk:**
- Протокол `TranscriptionModel` с реализациями: local, Groq, Deepgram, ElevenLabs, Gemini, Mistral
- `TranscriptionServiceRegistry` — маршрутизация по провайдеру
- Каталог моделей с метаданными (скорость, точность, RAM)

**Куда в Sheptun:** `recognition.py` — протокол `Recognizer` + реализации `WhisperRecognizer`, `GroqRecognizer`, `DeepgramRecognizer`.

**Сложность:** высокая

### 12. Streaming transcription

**Что делает VoiceInk:**
- Передача аудио-чанков в реальном времени для потоковой транскрипции
- Реализации для ElevenLabs, Deepgram, Mistral, Soniox
- Буферизация чанков до готовности сессии

**Куда в Sheptun:** альтернативный режим в `audio.py` + `recognition.py` для облачных API.

**Сложность:** высокая

---

## Архитектурные заметки из VoiceInk

### Параметры Whisper в VoiceInk
```
temperature = 0.2 (низкая для стабильности)
no_context = true (не использует контекст предыдущих сегментов)
flash_attn = true (Metal GPU ускорение)
threads = max(1, min(8, cpuCount - 2))
VAD: Silero v5 через whisper.cpp (threshold=0.50, min_speech=250ms, min_silence=100ms)
```

### Паттерны
- **State machine** для жизненного цикла: idle → starting → recording → transcribing → enhancing → busy
- **Параллельная загрузка модели** во время записи (Task.detached)
- **Dual callback** для streaming: буферизация чанков → переключение на реальный callback когда сессия готова
- **Fn-key debounce** 75мс из-за паразитных событий macOS
- **Cooldown** 0.5 сек для custom shortcuts от двойного срабатывания

### Каталог моделей VoiceInk
- Локальные: tiny, base, large-v2, large-v3, large-v3-turbo, large-v3-turbo-q5_0
- Облачные: Groq Whisper, ElevenLabs Scribe v1/v2, Deepgram Nova 3, Mistral Voxtral, Gemini 2.5/3, Soniox v4
