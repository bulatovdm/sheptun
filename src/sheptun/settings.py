import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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
    "Продолжение следует...",
    "Субтитры сделал DimaTorzworWorWorWork",
    "Субтитры создавал DimaTorzworWorWorWorks",
    "Субтитры делал DimaTorzworWorWorWorWorks",
    "Редактор субтитров А.Синецкая Корректор А.Егорова",
    "Спасибо за просмотр!",
    "Спасибо за внимание.",
    "Благодарю за внимание.",
    "До свидания.",
    "Пока.",
)


@dataclass(frozen=True)
class Settings:
    model: str = _get_str("SHEPTUN_MODEL", "base")
    device: str | None = _get_optional_str("SHEPTUN_DEVICE")
    energy_threshold: float = _get_float("SHEPTUN_ENERGY_THRESHOLD", 0.01)
    silence_duration: float = _get_float("SHEPTUN_SILENCE_DURATION", 0.5)
    min_speech_duration: float = _get_float("SHEPTUN_MIN_SPEECH_DURATION", 0.2)
    max_speech_duration: float = _get_float("SHEPTUN_MAX_SPEECH_DURATION", 30.0)
    vad_type: str = _get_str("SHEPTUN_VAD_TYPE", "energy")  # energy | silero
    hotkey_toggle: str = _get_str("SHEPTUN_HOTKEY_TOGGLE", "<ctrl>+<alt>+s")
    hotkey_ptt: str = _get_str("SHEPTUN_HOTKEY_PTT", "<ctrl>+<alt>+<space>")
    debug: bool = _get_bool("SHEPTUN_DEBUG", False)
    log_file: Path = Path(_get_str("SHEPTUN_LOG_FILE", "logs/sheptun.log"))
    app_path: Path = Path(_get_str("SHEPTUN_APP_PATH", "/Applications/Sheptun.app"))
    record_dataset: bool = _get_bool("SHEPTUN_RECORD_DATASET", False)
    dataset_path: Path = _get_path("SHEPTUN_DATASET_PATH", _DEFAULT_DATASET_PATH)
    hallucinations: tuple[str, ...] = _get_tuple("SHEPTUN_HALLUCINATIONS", _DEFAULT_HALLUCINATIONS)
    use_clipboard: bool = _get_bool("SHEPTUN_USE_CLIPBOARD", False)
    key_delay: float = _get_float("SHEPTUN_KEY_DELAY", 0.02)  # Low values may cause duplicates
    warmup_interval: float = _get_float("SHEPTUN_WARMUP_INTERVAL", 120.0)  # seconds, 0 to disable


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
