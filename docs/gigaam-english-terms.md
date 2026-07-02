# GigaAM и англоязычные термины: почему он не годится для Sheptun как есть

*Deep-research от 2 июля 2026. Узкий вопрос: справляется ли GigaAM-v3 с англицизмами (git, commit, Docker…) в русской речи.*
*Метод: 5 углов → 19 источников → 81 факт → 25 проверено (3-голосовая adversarial), 19 подтверждено, 6 опровергнуто.*

## Вердикт

**Наблюдение пользователя подтверждено — и это архитектурный потолок, а не недообученность.** GigaAM (включая v3) — **моноязычная русская** модель. Её символьная голова (char/CTC/RNNT) физически **не может выдать латиницу**, поэтому `git`, `commit`, `Docker` неизбежно превращаются в кириллическую фонетическую кашу. В v3 это **не изменилось**. Contextual biasing / hotwords в GigaAM **нет** (официально не в роадмапе). 

**Вывод: для смешанной ru/en технической речи Sheptun Whisper остаётся лучшим выбором**, несмотря на чуть худший русский WER — Whisper нативно выдаёт латиницу и поддерживает `initial_prompt`-биасинг.

## Подтверждённые факты (confidence: high)

### 1. GigaAM обучен ТОЛЬКО на русском (3-0 / 2-1)
Предобучение исключительно на русской речи (700K ч + 4K размеченных). В arXiv 2506.01192, HF-карточке и репозитории — **ноль** упоминаний code-switching, mixed-language, англицизмов. Модель позиционируется как моноязычный русский ASR.
Источники: [arXiv 2506.01192](https://arxiv.org/abs/2506.01192), [HF GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3), [habr 973160](https://habr.com/ru/articles/973160/), [GigaAM](https://github.com/salute-developers/GigaAM).

### 2. ⭐ Токенизатор — только кириллица (3-0, доказано кодом)
**Самый решающий факт.** Словарь char/rnnt генерируется как `['▁', *(chr(ord('а')+i) for i in range(32)), '<blk>']` = **34 токена** (пробел + а..я + blank), **ноль латинских букв**. e2e_rnnt — 1025-токенный русский BPE. Символьная голова **физически не может** вывести латиницу → корень «фонетической каши» для англотерминов.
Источники: [gigastt](https://github.com/ekhodzitsky/gigastt), [GigaAM](https://github.com/salute-developers/GigaAM), [HF GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3).

### 3. В GigaAM нет contextual biasing / hotwords (2-1)
Ни в README, ни в HF-карточке, ни в статье нет word-boosting/hotword/custom-vocab. Issue #31 «Are there any plans for Word Boosting?» (март 2025) — ответ мейнтейнера: contextual biasing «не в ближайшем роадмапе». Единственный внешний рычаг (в community-обёртке) — общий русский KenLM, но он биасит **кириллицу**, не латиницу.
Источники: [gigastt](https://github.com/ekhodzitsky/gigastt), [GigaAM](https://github.com/salute-developers/GigaAM), [gigaam-v3-ctc-with-lm](https://huggingface.co/waveletdeboshir/gigaam-v3-ctc-with-lm).

### 4. Если всё же брать GigaAM-CTC — только внешние обходы + словарь замен (3-0)
Английские термины можно тянуть лишь через decode-time тулинг поверх CTC-головы: **pyctcdecode hotwords**, **KenLM shallow fusion**, **NeMo CTC-WS / GPU Phrase-Boosting**. Модель `waveletdeboshir/gigaam-v3-ctc-with-lm` уже гоняет логиты GigaAM-CTC через pyctcdecode+KenLM (совместимость доказана). NeMo CTC-WS (arXiv 2406.07096) поднимает F-score контекстных слов с 0.32 до 0.87 без переобучения.
**⚠️ КРИТИЧНО:** все они бустят **кириллические** транскрипции («гит», «докер»), НЕ латиницу → всё равно нужен пост-словарь «каша→англотермин», **который у Sheptun уже есть (`replacements.yaml`)**.
Источники: [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode), [gigaam-v3-ctc-with-lm](https://huggingface.co/waveletdeboshir/gigaam-v3-ctc-with-lm), [arXiv 2406.07096](https://arxiv.org/html/2406.07096v1), [NeMo word_boosting](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html), [TurboBias 2508.07014](https://arxiv.org/abs/2508.07014).

### 5. Whisper выдаёт латиницу и поддерживает initial_prompt-биасинг (2-1 / 3-0)
Список целевых редких/техтерминов в prompt-контексте Whisper «значительно снижает WER» на них: zero-shot prompt-list биасинг снизил rare-word R-WER с 23.7% до 18.0% и OOV-WER с 60% до 37.1% на 11 датасетах ([2502.11572](https://arxiv.org/html/2502.11572v1)). Prefix-tree биасинг (TCPGen, [2410.18363](https://arxiv.org/abs/2410.18363)) направляет декодирование Whisper к кастомному словарю без правки весов.
**⚠️ Нюансы:** zero-shot выигрыш скромный и может чуть портить общий WER; prefix-tree требует обучить доп.модуль и проверялся только на английском (не на ru/en code-switching).

## Что это значит для Sheptun

| Вариант | Русский WER | Англотермины (git/commit/Docker) | Вывод |
|---|---|---|---|
| **Whisper (текущий)** | чуть хуже | **нативно латиницей** + initial_prompt-биасинг | ✅ Оставить основным для смешанной речи |
| **GigaAM-v3** | лучше | **невозможны нативно** (кириллица-only), кашей | ❌ Не подходит как замена без тяжёлых обходов |
| GigaAM-CTC + pyctcdecode/KenLM + replacements.yaml | лучше | кириллица-буст + пост-словарь → латиница | ⚠️ Сложно; выигрыш не доказан; только если очень нужен русский WER |

**Рекомендация:** остаёмся на Whisper. Практический шаг — **усилить `initial_prompt` англотерминами** (git, commit, PHP, Docker, Laravel, deploy, pull request…), это дешёвый рычаг, которого у GigaAM нет. GigaAM-v3 держать «на радаре» только если появится головной вывод с латиницей или официальный biasing.

## Caveats

1. **v3 vs общий фреймворк:** arXiv 2506.01192 описывает общий фреймворк/scaling GigaAM (100K ч в статье), «700K ч» — из HF-карточки v3. Оба русские — вывод держится, но не смешивать цифры.
2. **Противоречивые метаданные:** HF-карточка v3 в YAML тегает `language: ru, en` — но это **недокументированный голый тег** без бенчмарка, противоречащий доказанно кириллица-only char-голове. Трактовать как транслитерацию заимствований, не настоящий английский вывод.
3. **1025-BPE-голова** теоретически могла бы содержать редкие латинские сабворды (в дампах словаря не видно) — остаточная неопределённость, но документирована как русскоцентричная.
4. **Whisper-биасинг** проверялся на узком английском словаре, НЕ на ru/en code-switching — реальный выигрыш на терминале не доказан.
5. **Опровергнутые claims** (не выжили в проверке): что GigaAM выдаёт латиницу / сохраняет «YouTube»/«1941». Не полагаться.
6. **Нет прямого WER-бенчмарка** GigaAM vs Whisper на смешанных ru/en командах — вывод на архитектурном рассуждении, не на прямом замере.

## Открытые вопросы

1. Есть ли опубликованный/community-бенчмарк GigaAM-v3 vs Whisper именно на смешанных ru/en командах («запушь в main», «подними Docker контейнер»)?
2. Содержит ли 1025-BPE (e2e) голова латинские сабворды и выдаёт ли их на практике?
3. Route «GigaAM + Cyrillic→Latin словарь» vs «Whisper + initial_prompt» — что даёт лучшую end-to-end точность команд? Может ли пост-обработка когда-либо выиграть net?
4. Как T-one / Vosk / Silero справляются с англотерминами в русской речи — по этому под-вопросу подтверждённых фактов не нашлось, остаётся открытым.
