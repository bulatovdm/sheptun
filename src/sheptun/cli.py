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
            help="Модель Whisper (tiny, base, small, medium, large)",
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
            help="Модель Whisper (tiny, base, small, medium, large)",
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


@app.command()
def list_models() -> None:
    """Показать загруженные модели Whisper."""
    cache_dir = _get_whisper_cache_dir()
    if not cache_dir.exists():
        _hint("Кэш моделей пуст")
        return

    current_model = settings.model
    models = list(cache_dir.glob("*.pt"))

    if not models:
        _hint("Нет загруженных моделей")
        return

    console.print("\n[bold]Загруженные модели:[/bold]")
    for model_file in sorted(models):
        name = model_file.stem
        size_mb = model_file.stat().st_size / (1024 * 1024)
        is_active = name == current_model or name.startswith(f"{current_model}-")
        marker = " [green]← активная[/green]" if is_active else ""
        console.print(f"  {name} ({size_mb:.0f} MB){marker}")


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
