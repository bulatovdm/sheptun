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


def _get_hf_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _hf_repo_name(dir_name: str) -> str:
    return dir_name.removeprefix("models--").replace("--", "/")


def _get_active_hf_repos() -> set[str]:
    from sheptun.recognition import resolve_mlx_model

    repos: set[str] = set()
    if settings.recognizer == "mlx":
        repos.add(resolve_mlx_model(settings.model))
    if settings.recognizer == "qwen":
        repos.add("Qwen/Qwen3-ASR-0.6B")
    if settings.recognizer == "parakeet":
        repos.add("nvidia/parakeet-tdt-0.6b-v3")
    return repos


def _list_cached_models() -> list[tuple[str, str, float, bool]]:
    """List all cached models: (repo_name, dir_name, size_mb, is_active)."""
    active_repos = _get_active_hf_repos()
    active_whisper = settings.model if settings.recognizer == "whisper" else None
    results: list[tuple[str, str, float, bool]] = []

    # Whisper .pt files
    whisper_dir = _get_whisper_cache_dir()
    if whisper_dir.exists():
        for pt_file in sorted(whisper_dir.glob("*.pt")):
            name = pt_file.stem
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            is_active = active_whisper is not None and (
                name == active_whisper or name.startswith(f"{active_whisper}-")
            )
            results.append((f"whisper/{name}", pt_file.name, size_mb, is_active))

    # HuggingFace models
    hf_dir = _get_hf_cache_dir()
    if hf_dir.exists():
        for model_dir in sorted(hf_dir.glob("models--*")):
            if not model_dir.is_dir():
                continue
            repo = _hf_repo_name(model_dir.name)
            size_mb = _dir_size_mb(model_dir)
            is_active = repo in active_repos
            results.append((repo, model_dir.name, size_mb, is_active))

    return results


@app.command()
def cleanup_models(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Показать что будет удалено, без удаления",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Удалить без подтверждения",
        ),
    ] = False,
) -> None:
    """Удалить неиспользуемые модели из кэша."""
    models = _list_cached_models()
    if not models:
        _hint("Кэш моделей пуст")
        return

    inactive = [(repo, dir_name, size) for repo, dir_name, size, active in models if not active]
    if not inactive:
        _success("Нет неиспользуемых моделей")
        return

    total_mb = sum(size for _, _, size in inactive)
    console.print(f"\n[bold]Неиспользуемые модели ({total_mb:.0f} MB):[/bold]")
    for repo, _, size in inactive:
        console.print(f"  [red]✗[/red] {repo} ({size:.0f} MB)")

    if dry_run:
        _hint(f"Dry run: будет освобождено {total_mb:.0f} MB")
        return

    if not force:
        confirm = typer.confirm(f"\nУдалить {len(inactive)} моделей ({total_mb:.0f} MB)?")
        if not confirm:
            _hint("Отменено")
            return

    deleted_mb = 0.0
    whisper_dir = _get_whisper_cache_dir()
    hf_dir = _get_hf_cache_dir()

    for repo, dir_name, size in inactive:
        if repo.startswith("whisper/"):
            target = whisper_dir / dir_name
            target.unlink(missing_ok=True)
        else:
            import shutil

            target = hf_dir / dir_name
            shutil.rmtree(target, ignore_errors=True)
        deleted_mb += size
        _hint(f"Удалена: {repo} ({size:.0f} MB)")

    _success(f"Освобождено {deleted_mb:.0f} MB")


@app.command()
def list_models() -> None:
    """Показать загруженные модели и занимаемое место."""
    models = _list_cached_models()
    if not models:
        _hint("Нет загруженных моделей")
        return

    total_mb = sum(size for _, _, size, _ in models)
    active_mb = sum(size for _, _, size, active in models if active)
    inactive_mb = total_mb - active_mb

    console.print(f"\n[bold]Загруженные модели ({total_mb:.0f} MB):[/bold]")
    for repo, _, size, is_active in models:
        if is_active:
            console.print(f"  [green]✓[/green] {repo} ({size:.0f} MB) [green]← активная[/green]")
        else:
            console.print(f"  [dim]  {repo} ({size:.0f} MB)[/dim]")

    if inactive_mb > 100:
        console.print(
            f"\n[yellow]Можно освободить {inactive_mb:.0f} MB: "
            f"sheptun cleanup-models[/yellow]"
        )


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
def serve(
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Порт сервера"),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Токен авторизации"),
    ] = None,
) -> None:
    """Запустить сервер для приёма текста с удалённой машины."""
    from sheptun.keyboard import MacOSKeyboardSender
    from sheptun.remote import RemoteServer

    server_port = port or settings.remote_port
    server_token = token or settings.remote_token

    keyboard = MacOSKeyboardSender(use_clipboard=settings.use_clipboard)
    server = RemoteServer(
        keyboard_sender=keyboard,
        port=server_port,
        token=server_token,
    )

    _success(f"Сервер запущен на порту {server_port}")
    _hint("Нажмите Ctrl+C для остановки")
    server.start_blocking()


@app.command()
def remote_test(
    host: Annotated[
        str | None,
        typer.Argument(help="Хост удалённой машины (например, macbook.local)"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Порт сервера"),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Токен авторизации"),
    ] = None,
    text: Annotated[
        str,
        typer.Option("--text", "-t", help="Текст для отправки"),
    ] = "Привет от Шептуна!",
) -> None:
    """Проверить подключение к удалённому серверу."""
    from sheptun.remote import RemoteClient

    remote_host = host or settings.remote_host
    if not remote_host:
        _error("Укажите хост: sheptun remote-test macbook.local")
        _hint("Или задайте SHEPTUN_REMOTE_HOST в .env")
        raise typer.Exit(1)

    remote_port = port or settings.remote_port
    remote_token = token or settings.remote_token

    client = RemoteClient(host=remote_host, port=remote_port, token=remote_token)

    _info(f"Проверка подключения к {remote_host}:{remote_port}...")
    result = client.ping()
    if result is None:
        _error(f"Не удалось подключиться к {remote_host}:{remote_port}")
        raise typer.Exit(1)

    _success(f"Подключение установлено: {result.get('hostname', 'unknown')}")

    _info(f"Отправка текста: '{text}'")
    if client.send_text(text):
        _success("Текст отправлен")
    else:
        _error("Не удалось отправить текст")
        raise typer.Exit(1)


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


DEFAULT_TESTSET_DIR = Path("dataset/testset")
DEFAULT_BENCHMARK_MODELS = "mlx,whisper,apple,parakeet"
DEFAULT_BENCHMARK_FILES = 5


@app.command()
def benchmark(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Движки для сравнения через запятую: mlx,whisper,apple,parakeet",
        ),
    ] = DEFAULT_BENCHMARK_MODELS,
    files: Annotated[
        int,
        typer.Option(
            "--files",
            "-n",
            help="Количество аудиофайлов (0 = все)",
            min=0,
        ),
    ] = DEFAULT_BENCHMARK_FILES,
    audio_dir: Annotated[
        Path | None,
        typer.Option(
            "--audio-dir",
            "-d",
            help="Директория с .wav файлами",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    transcripts: Annotated[
        Path | None,
        typer.Option(
            "--transcripts",
            "-t",
            help="Файл с эталонными транскрипциями (.jsonl). По умолчанию ищется transcripts.jsonl рядом с аудио",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    testset: Annotated[
        bool,
        typer.Option(
            "--testset",
            help="Использовать эталонный тест-сет из dataset/testset/ (фиксированные фразы)",
        ),
    ] = False,
    no_refs: Annotated[
        bool,
        typer.Option(
            "--no-refs",
            help="Не использовать эталонные транскрипции (только скорость)",
        ),
    ] = False,
) -> None:
    """Сравнить скорость и качество распознавания разных движков."""
    from sheptun.benchmark import load_references, run_benchmark

    model_list = [m.strip() for m in models.split(",") if m.strip()]

    if testset:
        testset_dir = settings.dataset_path / "testset"
        wav_dir = audio_dir or testset_dir
        default_transcripts = testset_dir / "references.jsonl"
    else:
        wav_dir = audio_dir or (settings.dataset_path / "audio")
        default_transcripts = wav_dir.parent / "transcripts.jsonl"

    if not wav_dir.exists():
        _error(f"Директория не найдена: {wav_dir}")
        if testset:
            _hint("Запишите тест-сет: sheptun benchmark --testset --help")
        else:
            _hint("Укажите путь: --audio-dir /path/to/wavs")
        raise typer.Exit(1)

    audio_files = sorted(wav_dir.glob("*.wav"))
    if not audio_files:
        if testset:
            _error(f"Нет .wav файлов в {wav_dir}")
            _hint("Тест-сет содержит только references.jsonl. Запишите аудио для каждой фразы.")
            _hint(f"Фразы: {default_transcripts}")
        else:
            _error(f"Нет .wav файлов в {wav_dir}")
        raise typer.Exit(1)

    refs: dict[str, str] = {}
    if not no_refs:
        transcripts_path = transcripts or default_transcripts
        refs = load_references(transcripts_path)
        if refs:
            _hint(f"Эталоны загружены: {len(refs)} записей из {transcripts_path}")
        else:
            _hint("Эталонные транскрипции не найдены (только метрики скорости)")

    n_files = files if files > 0 else None
    run_benchmark(model_list, audio_files, n_files, references=refs)


_MIN_RECORD_DURATION = 0.3
_TESTSET_SAMPLE_RATE = 16000
_TESTSET_SAMPLE_WIDTH = 2
_TESTSET_CHANNELS = 1


def _save_testset_wav(path: Path, audio_bytes: bytes) -> float:
    """Сохранить аудиобуфер (int16 bytes) в .wav файл. Возвращает длительность в сек."""
    import wave

    import numpy as np

    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(audio_int16) / _TESTSET_SAMPLE_RATE
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(_TESTSET_CHANNELS)
        wf.setsampwidth(_TESTSET_SAMPLE_WIDTH)
        wf.setframerate(_TESTSET_SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return duration


def _wait_key(prompt: str, accepted: set[str] | None = None) -> str:
    """Ждать нажатия клавиши с подавлением ввода (не попадает в терминал/Sheptun).

    accepted — множество допустимых символов в нижнем регистре.
    Если None — принимает любой символ (включая Enter → "").
    Возвращает символ в нижнем регистре или "" для Enter/Space.
    """
    import threading

    from pynput import keyboard

    result: list[str] = []
    done = threading.Event()

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> bool | None:
        char = ""
        if key == keyboard.Key.enter or key == keyboard.Key.space:
            char = ""
        elif isinstance(key, keyboard.KeyCode) and key.char:
            char = key.char.lower()
        else:
            return None  # игнорировать служебные клавиши

        if accepted is None or char in accepted:
            result.append(char)
            done.set()
            return False  # остановить listener
        return None

    print(prompt, end="", flush=True)
    with keyboard.Listener(on_press=on_press, suppress=True):  # type: ignore[arg-type]
        done.wait()

    print()  # перенос строки после нажатия
    return result[0] if result else ""


def _record_phrase(phrase_id: str, text: str, note: str, out_path: Path) -> bool:
    """Интерактивная запись одной фразы. Возвращает True если сохранена, False если пропущена."""
    from sheptun.audio import AudioRecorder

    while True:
        console.print(f"\n  [bold]{text}[/bold]")
        if note:
            console.print(f"  [dim]({note})[/dim]")
        console.print()

        _wait_key("  Нажмите Enter или пробел чтобы начать запись... ")

        recorder = AudioRecorder()
        recorder.start()
        _wait_key("  ● Запись... нажмите Enter или пробел чтобы остановить")

        audio_bytes = recorder.stop()
        duration = len(audio_bytes) // (_TESTSET_SAMPLE_WIDTH * _TESTSET_CHANNELS) / _TESTSET_SAMPLE_RATE

        if duration < _MIN_RECORD_DURATION:
            _error(f"Слишком коротко ({duration:.1f} сек), попробуйте снова")
            continue

        console.print(f"  [green]✓ Записано {duration:.1f} сек[/green]")
        console.print()

        choice = _wait_key("  [Enter/пробел] сохранить  [r] перезаписать  [s] пропустить: ", {"", "r", "s"})

        if choice == "r":
            continue
        if choice == "s":
            _hint(f"  Пропущено: {phrase_id}")
            return False

        _save_testset_wav(out_path, audio_bytes)
        _success(f"Сохранено: {out_path.name}")
        return True


@app.command()
def record_testset(
    testset_dir: Annotated[
        Path | None,
        typer.Option(
            "--testset-dir",
            "-d",
            help="Директория тест-сета с references.jsonl",
            file_okay=False,
        ),
    ] = None,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help="Пропустить уже записанные файлы",
        ),
    ] = True,
    start_from: Annotated[
        int,
        typer.Option(
            "--start-from",
            "-s",
            help="Начать с указанной фразы (1-based)",
            min=1,
        ),
    ] = 1,
) -> None:
    """Интерактивная запись аудиофайлов для эталонного тест-сета."""
    from sheptun.benchmark import load_references

    tdir = testset_dir or (settings.dataset_path / "testset")
    refs_file = tdir / "references.jsonl"

    if not refs_file.exists():
        _error(f"Файл эталонов не найден: {refs_file}")
        raise typer.Exit(1)

    tdir.mkdir(parents=True, exist_ok=True)

    refs = load_references(refs_file)
    if not refs:
        _error("references.jsonl не содержит записей")
        raise typer.Exit(1)

    # Восстановить порядок и категории из jsonl напрямую
    import contextlib
    import json

    entries: list[dict[str, str]] = []
    with refs_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    entries.append(json.loads(line))

    total = len(entries)
    console.print(f"\n[bold]Запись тест-сета: {total} фраз[/bold]")
    console.print(f"Директория: {tdir}")
    if skip_existing:
        existing = sum(1 for e in entries if (tdir / f"{e['id']}.wav").exists())
        if existing:
            _hint(f"Уже записано: {existing}/{total}, будут пропущены")
    console.print()

    saved = 0
    skipped = 0

    for i, entry in enumerate(entries, start=1):
        if i < start_from:
            continue

        phrase_id = entry.get("id", "")
        text = entry.get("text", "")
        category = entry.get("category", "")
        note = entry.get("note", "")
        out_path = tdir / f"{phrase_id}.wav"

        if skip_existing and out_path.exists():
            _hint(f"[{i}/{total}] Пропущено (уже есть): {phrase_id}")
            skipped += 1
            continue

        console.print(f"[cyan][{i}/{total}][/cyan] [dim]{category}[/dim]")

        if _record_phrase(phrase_id, text, note, out_path):
            saved += 1
        else:
            skipped += 1

    console.print()
    _success(f"Готово: записано {saved}, пропущено {skipped} из {total}")
    if saved > 0:
        _hint("Запустите бенчмарк: sheptun benchmark --testset")


if __name__ == "__main__":
    app()
