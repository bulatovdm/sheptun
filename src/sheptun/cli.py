import signal
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sheptun import __version__
from sheptun.config import get_config_path
from sheptun.engine import VoiceEngine
from sheptun.settings import settings, setup_logging

app = typer.Typer(
    name="sheptun",
    help="Голосовое управление терминалом на русском языке",
    no_args_is_help=True,
)

console = Console()

PROCESS_KILL_DELAY = 0.5
APP_LAUNCH_DELAY = 1.0


def _success(msg: str) -> None:
    console.print(f"[green]✓ {msg}[/green]")


def _info(msg: str) -> None:
    console.print(f"[yellow]{msg}[/yellow]")


def _error(msg: str) -> None:
    console.print(f"[red]✗ {msg}[/red]")


def _hint(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


@app.command()
def listen(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Путь к файлу конфигурации команд",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Модель Whisper (tiny, base, small, medium, large, turbo)",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            "-d",
            help="Устройство для Whisper (cpu, cuda, mps)",
        ),
    ] = None,
    simple_status: Annotated[
        bool,
        typer.Option(
            "--simple",
            "-s",
            help="Использовать простой вывод статуса",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Выводить отладочную информацию",
        ),
    ] = False,
) -> None:
    """Запустить прослушивание голосовых команд."""
    use_debug = debug or settings.debug
    if use_debug:
        setup_logging(force=True)

    model_name = model or settings.model
    device_name = device or settings.device

    config_path = get_config_path(config)
    _hint(f"Конфигурация: {config_path}")
    _hint(f"Модель: {model_name}")
    if use_debug:
        _hint(f"Лог: {settings.log_file}")

    _info("Загрузка модели Whisper...")

    engine = VoiceEngine.create(
        config_path=config_path,
        model_name=model_name,
        device=device_name,
        use_live_status=not simple_status and not use_debug,
        debug=use_debug,
    )

    def signal_handler(_signum: int, _frame: object) -> None:
        _info("\nЗавершение работы...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _success("Модель загружена. Начинаю слушать...")
    _hint("Скажите 'стоп' или нажмите Ctrl+C для выхода\n")

    engine.start()

    try:
        while engine.is_running():
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


@app.command()
def version() -> None:
    """Показать версию приложения."""
    console.print(f"sheptun версия {__version__}")


@app.command()
def test_mic() -> None:
    """Проверить работу микрофона."""
    from typing import Any

    import numpy as np
    import sounddevice as sd

    _info("Проверка микрофона...")

    default_input: dict[str, Any] = sd.query_devices(kind="input")  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]

    console.print(f"\n[green]Устройство по умолчанию:[/green] {default_input['name']}")
    _hint(f"Каналы: {default_input['max_input_channels']}")
    _hint(f"Частота: {default_input['default_samplerate']} Hz")

    _info("\nЗапись 3 секунды...")

    try:
        recording: np.ndarray[Any, np.dtype[np.int16]] = sd.rec(  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]
            48000,
            samplerate=16000,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        energy: np.floating[Any] = np.sqrt(np.mean(recording.astype(np.float32) ** 2))

        if energy > 100:
            _success(f"Микрофон работает! Уровень сигнала: {energy:.0f}")
        else:
            _info(f"⚠ Слабый сигнал. Уровень: {energy:.0f}")

    except Exception as e:
        _error(f"Ошибка: {e}")
        raise typer.Exit(1) from None


@app.command()
def list_commands(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Путь к файлу конфигурации команд",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Показать список доступных команд."""
    from sheptun.commands import CommandConfigLoader

    config_path = get_config_path(config)
    command_config = CommandConfigLoader.load(config_path)

    console.print("\n[bold]Команды управления:[/bold]")
    for trigger, action in sorted(command_config.control_commands.items()):
        console.print(f"  [cyan]{trigger}[/cyan] → {action.action_type.name}: {action.value}")

    console.print("\n[bold]Команды остановки:[/bold]")
    for cmd in sorted(command_config.stop_commands):
        console.print(f"  [red]{cmd}[/red]")

    console.print("\n[bold]Slash-команды (слэш <команда>):[/bold]")
    for trigger, value in sorted(command_config.slash_commands.items()):
        console.print(f"  [yellow]слэш {trigger}[/yellow] → {value}")

    console.print("\n[bold]Префиксы диктовки:[/bold]")
    for prefix in command_config.dictation_prefixes:
        console.print(f"  [green]{prefix}[/green] <текст>")


@app.command()
def menubar(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Модель Whisper (tiny, base, small, medium, large, turbo)",
        ),
    ] = None,
) -> None:
    """Запустить как menubar приложение."""
    from sheptun.menubar import run_menubar

    model_name = model or settings.model
    _info("Запуск menubar приложения...")
    run_menubar(model_name=model_name)


def _kill_menubar_app() -> None:
    import subprocess
    import time

    _info("Закрытие Sheptun...")
    subprocess.run(["pkill", "-f", "sheptun.menubar"], check=False)
    time.sleep(PROCESS_KILL_DELAY)


def _launch_menubar_app() -> None:
    import subprocess
    import time

    app_path = settings.app_path
    if not app_path.exists():
        _error(f"Приложение не найдено: {app_path}")
        _hint("Запустите 'sheptun install-app' для установки")
        return

    _info("Запуск Sheptun...")
    subprocess.run(
        ["open", str(app_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    time.sleep(APP_LAUNCH_DELAY)


@app.command()
def restart() -> None:
    """Перезапустить menubar приложение."""
    _kill_menubar_app()
    _launch_menubar_app()
    _success("Sheptun перезапущен")


def _preload_whisper_model() -> None:
    if settings.recognizer == "apple":
        _info("Используется Apple Speech Framework (без загрузки Whisper)")
        return

    import whisper

    model_name = settings.model
    _info(f"Загрузка модели Whisper '{model_name}'...")
    whisper.load_model(model_name)
    _success(f"Модель Whisper '{model_name}' загружена")


def _preload_spelling_model() -> None:
    from sheptun.spelling import SpellCorrectorType, download_model

    spell_type = settings.spell_correction
    if spell_type == SpellCorrectorType.NONE.value:
        return

    _info(f"Загрузка модели коррекции '{spell_type}'...")
    try:
        download_model()
        _success(f"Модель коррекции '{spell_type}' загружена")
    except ImportError:
        _hint("Коррекция орфографии недоступна (установите: pip install -e '.[spelling]')")


def _get_whisper_cache_dir() -> Path:
    return Path.home() / ".cache" / "whisper"


@app.command()
def cleanup_models(
    model: Annotated[
        str | None,
        typer.Argument(help="Имя модели для удаления (или все неактивные если не указано)"),
    ] = None,
) -> None:
    """Удалить модели Whisper."""
    cache_dir = _get_whisper_cache_dir()
    if not cache_dir.exists():
        _hint("Кэш моделей пуст")
        return

    current_model = settings.model

    if model:
        if model == current_model:
            _error(f"Нельзя удалить активную модель '{model}'")
            raise typer.Exit(1)

        model_file = cache_dir / f"{model}.pt"
        if not model_file.exists():
            _error(f"Модель '{model}' не найдена")
            raise typer.Exit(1)

        size_mb = model_file.stat().st_size / (1024 * 1024)
        model_file.unlink()
        _success(f"Удалена модель '{model}' ({size_mb:.0f} MB)")
        return

    deleted_count = 0
    total_size_mb = 0.0

    def is_active_model(filename: str) -> bool:
        name = filename.removesuffix(".pt")
        return name == current_model or name.startswith(f"{current_model}-")

    for model_file in cache_dir.glob("*.pt"):
        if not is_active_model(model_file.name):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            model_file.unlink()
            _hint(f"Удалена: {model_file.name} ({size_mb:.0f} MB)")
            deleted_count += 1
            total_size_mb += size_mb

    if deleted_count == 0:
        _success("Нет неиспользуемых моделей")
    else:
        _success(f"Удалено {deleted_count} моделей ({total_size_mb:.0f} MB)")


def _get_mlx_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def _list_mlx_models() -> list[tuple[str, float]]:
    cache_dir = _get_mlx_cache_dir()
    if not cache_dir.exists():
        return []

    results: list[tuple[str, float]] = []
    for model_dir in sorted(cache_dir.glob("models--mlx-community--whisper-*")):
        name = model_dir.name.removeprefix("models--mlx-community--").replace("--", "/")
        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        results.append((name, size_bytes / (1024 * 1024)))
    return results


@app.command()
def list_models() -> None:
    """Показать загруженные модели Whisper."""
    current_model = settings.model
    current_recognizer = settings.recognizer
    has_any = False

    cache_dir = _get_whisper_cache_dir()
    whisper_models = list(cache_dir.glob("*.pt")) if cache_dir.exists() else []

    if whisper_models:
        has_any = True
        console.print("\n[bold]Whisper модели:[/bold]")
        for model_file in sorted(whisper_models):
            name = model_file.stem
            size_mb = model_file.stat().st_size / (1024 * 1024)
            is_active = current_recognizer == "whisper" and (
                name == current_model or name.startswith(f"{current_model}-")
            )
            marker = " [green]← активная[/green]" if is_active else ""
            console.print(f"  {name} ({size_mb:.0f} MB){marker}")

    mlx_models = _list_mlx_models()
    if mlx_models:
        has_any = True
        console.print("\n[bold]MLX Whisper модели:[/bold]")
        for name, size_mb in mlx_models:
            is_active = current_recognizer == "mlx" and current_model in name
            marker = " [green]← активная[/green]" if is_active else ""
            console.print(f"  {name} ({size_mb:.0f} MB){marker}")

    if not has_any:
        _hint("Нет загруженных моделей")


@app.command()
def install_app(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Путь для установки приложения",
        ),
    ] = None,
) -> None:
    """Создать macOS приложение (.app) для menubar."""
    from sheptun.app_builder import build_app

    _kill_menubar_app()
    _preload_whisper_model()
    _preload_spelling_model()

    app_dir = output or settings.app_path
    build_app(app_dir)

    _success(f"Приложение создано: {app_dir}")
    _hint("Теперь можно запустить из Launchpad или Finder")


@app.command()
def download_spelling() -> None:
    """Загрузить модель для коррекции орфографии."""
    from sheptun.spelling import SpellCorrectorType, download_model

    spell_type = settings.spell_correction
    if spell_type == SpellCorrectorType.NONE.value:
        _hint("Коррекция орфографии отключена (spell_correction=none)")
        return

    _info(f"Загрузка модели '{spell_type}'...")
    try:
        download_model()
        _success("Модель загружена")
    except ImportError as e:
        _error(f"Не установлены зависимости: {e}")
        _hint("Выполните: pip install -e '.[spelling]'")
        raise typer.Exit(1) from None
    except Exception as e:
        _error(f"Ошибка загрузки: {e}")
        raise typer.Exit(1) from None


@app.command()
def clear_dataset(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Пропустить подтверждение",
        ),
    ] = False,
) -> None:
    """Очистить датасет для fine-tuning."""
    from sheptun.dataset import DatasetRecorder

    recorder = DatasetRecorder()
    stats = recorder.get_stats()

    if stats["audio_files"] == 0:
        _hint("Датасет пуст")
        return

    console.print(f"\n[bold]Датасет:[/bold] {settings.dataset_path}")
    console.print(f"  Аудио файлов: {stats['audio_files']}")
    console.print(f"  Транскрипций: {stats['transcripts']}\n")

    if not force:
        confirm = typer.confirm("Удалить все данные?", default=False)
        if not confirm:
            _hint("Отменено")
            return

    recorder.clear()
    _success("Датасет очищен")


@app.command()
def verify_dataset(
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Количество записей для обработки",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Модель Claude (по умолчанию из Agent SDK)",
        ),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Путь к датасету",
        ),
    ] = None,
    retry: Annotated[
        bool,
        typer.Option(
            "--retry",
            "-r",
            help="Повторить обработку записей с ошибками",
        ),
    ] = False,
    reset: Annotated[
        bool,
        typer.Option(
            "--reset",
            help="Сбросить все результаты и обработать заново",
        ),
    ] = False,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-j",
            help="Количество параллельных запросов к Claude",
            min=1,
            max=10,
        ),
    ] = 1,
) -> None:
    """Верифицировать транскрипции через Claude."""
    import anyio

    from sheptun.verification import VerificationDB, run_verification

    ds_path = dataset or settings.dataset_path
    db_path = ds_path / "verification.db"

    if (retry or reset) and db_path.exists():
        db = VerificationDB(db_path)
        try:
            count = db.reset_all() if reset else db.reset_errors()
        finally:
            db.close()
        if count > 0:
            _info(f"Сброшено {count} записей для повторной обработки")
        else:
            _hint("Нет записей для сброса")
            if not reset:
                return

    model_name = model or settings.verify_model
    jobs = concurrency if concurrency > 1 else settings.verify_concurrency
    anyio.run(run_verification, dataset, limit, model_name, jobs)


@app.command()
def verify_status(
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Путь к датасету",
        ),
    ] = None,
) -> None:
    """Показать статус верификации транскрипций."""
    from sheptun.verification import VerificationDB

    ds_path = dataset or settings.dataset_path
    db_path = ds_path / "verification.db"

    if not db_path.exists():
        _hint("База верификации не найдена. Запустите verify-dataset.")
        return

    db = VerificationDB(db_path)
    try:
        stats = db.get_stats()
    finally:
        db.close()

    console.print(f"\n[bold]Верификация транскрипций:[/bold] {db_path}")
    console.print(f"  Всего: {stats.get('total', 0)}")
    console.print(f"  Ожидают: [yellow]{stats.get('pending', 0)}[/yellow]")
    console.print(f"  Обработано: [green]{stats.get('completed', 0)}[/green]")
    console.print(f"    Верных: {stats.get('correct', 0)}")
    console.print(f"    Исправлено: {stats.get('fixed', 0)}")
    console.print(f"    Галлюцинаций: [red]{stats.get('hallucinations', 0)}[/red]")
    console.print(f"  Ошибок: [red]{stats.get('error', 0)}[/red]")


@app.command()
def verify_export(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Путь к выходному файлу",
        ),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Путь к датасету",
        ),
    ] = None,
    include_hallucinations: Annotated[
        bool,
        typer.Option(
            "--include-hallucinations",
            help="Включить галлюцинации в экспорт",
        ),
    ] = False,
) -> None:
    """Экспортировать верифицированные транскрипции в JSONL."""
    from sheptun.verification import VerificationDB

    ds_path = dataset or settings.dataset_path
    db_path = ds_path / "verification.db"

    if not db_path.exists():
        _error("База верификации не найдена. Запустите verify-dataset.")
        raise typer.Exit(1)

    output_path = output or (ds_path / "transcripts_verified.jsonl")

    db = VerificationDB(db_path)
    try:
        count = db.export_jsonl(output_path, exclude_hallucinations=not include_hallucinations)
    finally:
        db.close()

    if count == 0:
        _hint("Нет обработанных записей для экспорта")
    else:
        _success(f"Экспортировано {count} записей в {output_path}")


@app.command()
def finetune_prepare(
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", "-d", help="Путь к датасету"),
    ] = None,
    min_confidence: Annotated[
        str | None,
        typer.Option("--min-confidence", help="Минимальный confidence (low, medium, high)"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Базовая модель Whisper (tiny, base, small, medium, large)"
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Путь для сохранения"),
    ] = None,
) -> None:
    """Подготовить датасет для fine-tuning из verification.db."""
    from sheptun.finetune import config_from_settings, prepare_dataset

    overrides: dict[str, object] = {}
    if dataset:
        overrides["dataset"] = dataset
    if min_confidence:
        overrides["min_confidence"] = min_confidence
    if model:
        overrides["model"] = model
    if output:
        overrides["output"] = output

    config = config_from_settings(**overrides)

    _info(f"Подготовка датасета из {config.dataset_path / 'verification.db'}")
    _info(f"Базовая модель: {config.base_model}")
    _info(f"Минимальный confidence: {config.min_confidence}")

    try:
        stats = prepare_dataset(config)
    except FileNotFoundError as e:
        _error(str(e))
        raise typer.Exit(1) from None
    except ValueError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    _success(
        f"Датасет подготовлен: {stats['total']} записей "
        f"(train: {stats['train']}, eval: {stats['eval']})"
    )
    _info(f"Сохранено в {config.output_dir / 'dataset'}")


@app.command()
def finetune_train(
    method: Annotated[
        str | None,
        typer.Option("--method", help="Метод обучения (lora, full)"),
    ] = None,
    steps: Annotated[
        int | None,
        typer.Option("--steps", help="Количество шагов обучения"),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Размер батча"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Базовая модель Whisper"),
    ] = None,
    lr: Annotated[
        float | None,
        typer.Option("--lr", help="Learning rate"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Путь для сохранения модели"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Продолжить обучение с последнего checkpoint"),
    ] = False,
) -> None:
    """Запустить fine-tuning модели Whisper."""
    from sheptun.finetune import config_from_settings, train

    overrides: dict[str, object] = {}
    if method:
        overrides["method"] = method
    if steps:
        overrides["steps"] = steps
    if batch_size:
        overrides["batch_size"] = batch_size
    if model:
        overrides["model"] = model
    if lr:
        overrides["lr"] = lr
    if output:
        overrides["output"] = output

    config = config_from_settings(**overrides)

    _info(f"Модель: {config.base_model}")
    _info(f"Метод: {config.method}")
    _info(f"Шаги: {config.max_steps}, batch: {config.batch_size}, lr: {config.learning_rate}")

    try:
        result_path = train(config, resume=resume)
    except FileNotFoundError as e:
        _error(str(e))
        _hint("Сначала подготовьте датасет: sheptun finetune-prepare")
        raise typer.Exit(1) from None
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            _error(f"Нехватка памяти: {e}")
            _hint("Попробуйте: --method lora --batch-size 2")
        else:
            _error(str(e))
        raise typer.Exit(1) from None

    _success(f"Модель сохранена: {result_path}")
    _hint(f"Для использования: SHEPTUN_MODEL={result_path} sheptun listen")


@app.command()
def finetune_eval(
    model_path: Annotated[
        Path | None,
        typer.Option("--model-path", help="Путь к fine-tuned модели"),
    ] = None,
    base_model: Annotated[
        str | None,
        typer.Option("--base-model", help="Базовая модель для сравнения"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Путь к директории fine-tuning"),
    ] = None,
) -> None:
    """Оценить качество fine-tuned модели (WER/CER)."""
    from sheptun.finetune import config_from_settings, evaluate

    overrides: dict[str, object] = {}
    if base_model:
        overrides["model"] = base_model
    if output:
        overrides["output"] = output
    if model_path:
        overrides["output"] = model_path

    config = config_from_settings(**overrides)

    _info(f"Оценка модели: {config.output_dir}")
    _info(f"Базовая модель: {config.base_model}")

    try:
        results = evaluate(config)
    except FileNotFoundError as e:
        _error(str(e))
        raise typer.Exit(1) from None

    console.print()
    console.print("[bold]Результаты оценки:[/bold]")
    console.print(f"  {'Метрика':<10} {'Base':<12} {'Fine-tuned':<12} {'Δ':<10}")
    console.print(f"  {'─' * 44}")
    for metric in ("wer", "cer"):
        base_val = results[f"{metric}_base"]
        ft_val = results[f"{metric}_finetuned"]
        delta = ft_val - base_val
        sign = "+" if delta > 0 else ""
        console.print(
            f"  {metric.upper():<10} {base_val:<12.2%} {ft_val:<12.2%} {sign}{delta:<10.2%}"
        )
    console.print()


@app.command()
def enable_autostart() -> None:
    """Включить автозапуск Sheptun при старте системы."""
    import subprocess

    app_path = settings.app_path
    if not app_path.exists():
        _error(f"Приложение не найдено: {app_path}")
        _hint("Сначала установите приложение: sheptun install-app")
        raise typer.Exit(1)

    script = f"""
        tell application "System Events"
            make login item at end with properties {{path:"{app_path}", hidden:false}}
        end tell
    """

    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        _success("Автозапуск включен")
        _hint("Sheptun будет запускаться при входе в систему")
    except subprocess.CalledProcessError as e:
        _error(f"Ошибка: {e.stderr}")
        raise typer.Exit(1) from e


@app.command()
def disable_autostart() -> None:
    """Отключить автозапуск Sheptun."""
    import subprocess

    script = """
        tell application "System Events"
            delete login item "Sheptun"
        end tell
    """

    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        _success("Автозапуск отключен")
    except subprocess.CalledProcessError as e:
        if "Can't get login item" in e.stderr:
            _hint("Автозапуск не был настроен")
        else:
            _error(f"Ошибка: {e.stderr}")
            raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
