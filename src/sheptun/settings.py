import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from sheptun.prompts import load_prompt

load_dotenv()


def _get_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes")


def _get_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    return float(value)


def _get_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_optional_str(key: str) -> str | None:
    value = os.getenv(key)
    return value if value else None


def _get_path(key: str, default: Path) -> Path:
    return Path(_get_str(key, str(default))).expanduser()


def _get_tuple(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(key)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


_DEFAULT_DATASET_PATH = Path("dataset")

_DEFAULT_HALLUCINATIONS = (
    # YouTube-style endings (Russian)
    "Продолжение следует...",
    "Спасибо за просмотр",
    "Спасибо за внимание",
    "Благодарю за внимание",
    "Подписывайтесь на канал",
    "Подписывайтесь на мой канал",
    "Ставьте лайки",
    "До новых встреч",
    # Subtitle credits (Russian)
    "Спасибо за субтитры Алексею Дубровскому",
    "Субтитры сделал",
    "Субтитры создавал",
    "Субтитры делал",
    "Редактор субтитров",
    "Корректор А.",
    "А.Егорова",
    "А.Семкин",
    "А.Синецкая",
    "DimaTorzok",
    # YouTube-style endings (English - may appear in Russian audio)
    "Thank you for watching",
    "Thanks for watching",
    "See you in the next video",
    "Subscribe to my channel",
    "Like and subscribe",
    # Subtitle credits (multilingual)
    "Amara.org",
    "Sottotitoli",
    "Legendas",
    "Untertitel",
    # Music/sound markers
    "♪",
    "♫",
)


@dataclass(frozen=True)
class Settings:
    model: str = _get_str("SHEPTUN_MODEL", "base")
    device: str | None = _get_optional_str("SHEPTUN_DEVICE")
    energy_threshold: float = _get_float("SHEPTUN_ENERGY_THRESHOLD", 0.01)
    silence_duration: float = _get_float("SHEPTUN_SILENCE_DURATION", 0.5)
    min_speech_duration: float = _get_float("SHEPTUN_MIN_SPEECH_DURATION", 0.2)
    max_speech_duration: float = _get_float("SHEPTUN_MAX_SPEECH_DURATION", 30.0)
    idle_timeout: float = _get_float("SHEPTUN_IDLE_TIMEOUT", 5.0)  # Reset buffer after silence
    vad_type: str = _get_str("SHEPTUN_VAD_TYPE", "energy")  # energy | silero
    hotkey_toggle: str = _get_str("SHEPTUN_HOTKEY_TOGGLE", "<ctrl>+<alt>+s")
    hotkey_ptt: str = _get_str("SHEPTUN_HOTKEY_PTT", "<ctrl>+<alt>+<space>")
    debug: bool = _get_bool("SHEPTUN_DEBUG", False)
    log_file: Path = Path(_get_str("SHEPTUN_LOG_FILE", "logs/sheptun.log"))
    app_path: Path = Path(_get_str("SHEPTUN_APP_PATH", "/Applications/Sheptun.app"))
    record_dataset: bool = _get_bool("SHEPTUN_RECORD_DATASET", False)
    dataset_path: Path = _get_path("SHEPTUN_DATASET_PATH", _DEFAULT_DATASET_PATH)
    hallucinations: tuple[str, ...] = _get_tuple("SHEPTUN_HALLUCINATIONS", _DEFAULT_HALLUCINATIONS)
    # Clipboard paste (Cmd+V) is atomic and avoids the character-duplication that
    # synthesized key events cause in Electron terminals (VS Code). The paste is
    # marked concealed/transient so it stays out of clipboard-manager history.
    use_clipboard: bool = _get_bool("SHEPTUN_USE_CLIPBOARD", True)
    key_delay: float = _get_float("SHEPTUN_KEY_DELAY", 0.02)  # Low values may cause duplicates
    warmup_interval: float = _get_float("SHEPTUN_WARMUP_INTERVAL", 120.0)  # seconds, 0 to disable
    auto_space: bool = _get_bool("SHEPTUN_AUTO_SPACE", True)  # Add trailing space to text
    # Spell correction: none, t5-russian (200M)
    spell_correction: str = _get_str("SHEPTUN_SPELL_CORRECTION", "none")
    # Whisper initial_prompt: context hint for better recognition of domain terms.
    # Default text lives in prompts/whisper_initial.md; override via env.
    initial_prompt: str = _get_str("SHEPTUN_INITIAL_PROMPT", load_prompt("whisper_initial"))
    # Recognizer: whisper, apple, mlx, parakeet, qwen
    recognizer: str = _get_str("SHEPTUN_RECOGNIZER", "whisper")
    # Apple Speech locale: ru-RU, en-US, etc
    apple_locale: str = _get_str("SHEPTUN_APPLE_LOCALE", "ru-RU")
    # Remote text delivery
    remote_enabled: bool = _get_bool("SHEPTUN_REMOTE_ENABLED", False)
    remote_serve: bool = _get_bool("SHEPTUN_REMOTE_SERVE", False)
    remote_host: str | None = _get_optional_str("SHEPTUN_REMOTE_HOST")
    remote_port: int = int(_get_float("SHEPTUN_REMOTE_PORT", 7849))
    remote_token: str = _get_str("SHEPTUN_REMOTE_TOKEN", "")
    remote_auto_detect: bool = _get_bool("SHEPTUN_REMOTE_AUTO_DETECT", True)
    # Fine-tuning
    finetune_model: str = _get_str("SHEPTUN_FINETUNE_MODEL", "large")
    finetune_method: str = _get_str("SHEPTUN_FINETUNE_METHOD", "lora")
    finetune_steps: int = int(_get_float("SHEPTUN_FINETUNE_STEPS", 4000))
    finetune_batch_size: int = int(_get_float("SHEPTUN_FINETUNE_BATCH_SIZE", 4))
    finetune_lr: float = _get_float("SHEPTUN_FINETUNE_LR", 1e-5)
    finetune_warmup_steps: int = int(_get_float("SHEPTUN_FINETUNE_WARMUP_STEPS", 500))
    finetune_output: Path = _get_path("SHEPTUN_FINETUNE_OUTPUT", Path("models/whisper-sheptun"))
    finetune_min_confidence: str = _get_str("SHEPTUN_FINETUNE_MIN_CONFIDENCE", "medium")
    # Anthropic SDK (log analyzer). Custom base_url/api_key from env.
    anthropic_base_url: str | None = _get_optional_str("SHEPTUN_ANTHROPIC_BASE_URL")
    anthropic_api_key: str | None = _get_optional_str("SHEPTUN_ANTHROPIC_API_KEY")
    # Log analyzer for replacement suggestions
    analyzer_model: str = _get_str("SHEPTUN_ANALYZER_MODEL", "claude-opus-4-8")
    analyzer_context_lines: int = int(_get_float("SHEPTUN_ANALYZER_CONTEXT_LINES", 10))
    # Context windows packed into one model request. Kept small: at 20 the answer overruns
    # max_tokens=8000 (truncated JSON, dropped rules, ~42s) — at 8 it completes cleanly (end_turn,
    # ~26s). Fewer windows per request → shorter answer → faster and no truncation on the proxy.
    analyzer_batch_size: int = int(_get_float("SHEPTUN_ANALYZER_BATCH_SIZE", 8))
    analyzer_max_windows: int = int(_get_float("SHEPTUN_ANALYZER_MAX_WINDOWS", 0))  # 0 = no limit
    analyzer_min_freq: int = int(_get_float("SHEPTUN_ANALYZER_MIN_FREQ", 1))
    analyzer_effort: str = _get_str("SHEPTUN_ANALYZER_EFFORT", "medium")
    # Output cap per request. At 8000 a dense batch occasionally overruns and truncates the JSON
    # (stop=max_tokens, dropped rules); 12000 leaves headroom so even rule-heavy batches finish.
    analyzer_max_tokens: int = int(_get_float("SHEPTUN_ANALYZER_MAX_TOKENS", 12000))
    # Extended thinking (reasoning). Off by default: with thinking on, Sonnet spends most of
    # its output budget on reasoning tokens, hitting max_tokens (truncated JSON) and taking
    # ~3x longer per batch. Off → shorter, faster, complete answers. effort only applies when on.
    analyzer_thinking: bool = _get_bool("SHEPTUN_ANALYZER_THINKING", False)
    # Minimum confidence to keep a suggestion: low | medium | high
    analyzer_min_confidence: str = _get_str("SHEPTUN_ANALYZER_MIN_CONFIDENCE", "medium")
    # Max model requests (batches) per run — 0 = unlimited. Bounds cost/time per run.
    analyzer_max_iterations: int = int(_get_float("SHEPTUN_ANALYZER_MAX_ITERATIONS", 0))
    # Second verification pass: a critic call re-checks candidates (extra request per batch).
    analyzer_verify: bool = _get_bool("SHEPTUN_ANALYZER_VERIFY", True)
    # Stream model responses (SSE) so a long generation keeps the connection alive and
    # doesn't hit a proxy read-timeout (Cloudflare 524). Same result, different transport.
    analyzer_stream: bool = _get_bool("SHEPTUN_ANALYZER_STREAM", True)
    # Send the list of already-covered `old` keys into every prompt so the model skips them.
    # Off by default: with 1000+ rules that block is ~4-5k tokens per request; dedup against
    # existing rules still runs before writing, so dropping it costs no correctness, only tokens.
    analyzer_send_known: bool = _get_bool("SHEPTUN_ANALYZER_SEND_KNOWN", False)
    # How many batch requests to run in parallel. 1 = sequential. The proxy handles ~5 concurrent
    # requests without latency degradation (measured); above ~8 the tail latency grows. The saved
    # checkpoint advances only over the contiguous completed prefix, so a crash never skips a batch.
    analyzer_concurrency: int = int(_get_float("SHEPTUN_ANALYZER_CONCURRENCY", 5))
    # SDK-level retries per request — 0 = fail fast; our own backoff loop handles retries instead.
    analyzer_max_retries: int = int(_get_float("SHEPTUN_ANALYZER_MAX_RETRIES", 0))
    # Pause (seconds) between successful model requests — eases load on the proxy/origin (502s).
    analyzer_delay: float = _get_float("SHEPTUN_ANALYZER_DELAY", 0.5)
    # On a request error: wait backoff*attempt seconds (15, 30, 45, …) and retry the same batch.
    analyzer_retry_backoff: float = _get_float("SHEPTUN_ANALYZER_RETRY_BACKOFF", 15.0)
    # How many times to retry a failing batch before giving up and stopping the run.
    analyzer_max_error_retries: int = int(_get_float("SHEPTUN_ANALYZER_MAX_ERROR_RETRIES", 4))
    # Override User-Agent — some proxies block the default Anthropic SDK UA (empty = SDK default)
    analyzer_user_agent: str | None = _get_optional_str("SHEPTUN_ANALYZER_USER_AGENT")
    # Log each request's user prompt and the model's raw reply (DEBUG level) so a malformed
    # or unparsable answer is visible in the log. Off by default — the prompts are large and
    # would bloat the log across thousands of requests. Turn on to debug parsing/empty replies.
    analyzer_log_prompts: bool = _get_bool("SHEPTUN_ANALYZER_LOG_PROMPTS", False)


settings = Settings()


def setup_logging(force: bool = False) -> None:
    if not settings.debug and not force:
        return

    import logging

    log_file = settings.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )
