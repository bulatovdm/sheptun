import logging
import signal
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sheptun import __version__
from sheptun.config import get_config_path
from sheptun.engine import VoiceEngine

LOG_FILE = Path.cwd() / "logs" / "sheptun.log"

app = typer.Typer(
    name="sheptun",
    help="Голосовое управление терминалом на русском языке",
    no_args_is_help=True,
)

console = Console()


def setup_debug_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


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
        str,
        typer.Option(
            "--model",
            "-m",
            help="Модель Whisper (tiny, base, small, medium, large)",
        ),
    ] = "base",
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
    if debug:
        setup_debug_logging()

    config_path = get_config_path(config)
    console.print(f"[dim]Конфигурация: {config_path}[/dim]")
    console.print(f"[dim]Модель: {model}[/dim]")
    if debug:
        console.print(f"[dim]Лог: {LOG_FILE}[/dim]")

    console.print("[yellow]Загрузка модели Whisper...[/yellow]")

    engine = VoiceEngine.create(
        config_path=config_path,
        model_name=model,
        device=device,
        use_live_status=not simple_status and not debug,
        debug=debug,
    )

    def signal_handler(_signum: int, _frame: object) -> None:
        console.print("\n[yellow]Завершение работы...[/yellow]")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    console.print("[green]Модель загружена. Начинаю слушать...[/green]")
    console.print("[dim]Скажите 'стоп' или нажмите Ctrl+C для выхода[/dim]\n")

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

    console.print("[yellow]Проверка микрофона...[/yellow]")

    default_input: dict[str, Any] = sd.query_devices(kind="input")  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]

    console.print(f"\n[green]Устройство по умолчанию:[/green] {default_input['name']}")
    console.print(f"[dim]Каналы: {default_input['max_input_channels']}[/dim]")
    console.print(f"[dim]Частота: {default_input['default_samplerate']} Hz[/dim]")

    console.print("\n[yellow]Запись 3 секунды...[/yellow]")

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
            console.print(f"[green]✓ Микрофон работает! Уровень сигнала: {energy:.0f}[/green]")
        else:
            console.print(f"[yellow]⚠ Слабый сигнал. Уровень: {energy:.0f}[/yellow]")

    except Exception as e:
        console.print(f"[red]✗ Ошибка: {e}[/red]")
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
        str,
        typer.Option(
            "--model",
            "-m",
            help="Модель Whisper (tiny, base, small, medium, large)",
        ),
    ] = "base",
) -> None:
    """Запустить как menubar приложение."""
    from sheptun.menubar import run_menubar

    console.print("[yellow]Запуск menubar приложения...[/yellow]")
    run_menubar(model_name=model)


if __name__ == "__main__":
    app()
