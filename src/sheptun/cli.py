import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from PIL import Image

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
    console.print(f"[dim]Конфигурация: {config_path}[/dim]")
    console.print(f"[dim]Модель: {model_name}[/dim]")
    if use_debug:
        console.print(f"[dim]Лог: {settings.log_file}[/dim]")

    console.print("[yellow]Загрузка модели Whisper...[/yellow]")

    engine = VoiceEngine.create(
        config_path=config_path,
        model_name=model_name,
        device=device_name,
        use_live_status=not simple_status and not use_debug,
        debug=use_debug,
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
    console.print("[yellow]Запуск menubar приложения...[/yellow]")
    run_menubar(model_name=model_name)


def _kill_menubar_app() -> None:
    import subprocess

    console.print("[yellow]Закрытие Sheptun...[/yellow]")
    subprocess.run(["pkill", "-f", "sheptun.menubar"], check=False)


def _launch_menubar_app() -> None:
    import subprocess

    console.print("[yellow]Запуск Sheptun...[/yellow]")
    subprocess.Popen(
        ["open", str(settings.app_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


@app.command()
def restart() -> None:
    """Перезапустить menubar приложение."""
    _kill_menubar_app()
    _launch_menubar_app()
    console.print("[green]✓ Sheptun перезапущен[/green]")


def _preload_whisper_model() -> None:
    import whisper

    model_name = settings.model
    console.print(f"[yellow]Загрузка модели Whisper '{model_name}'...[/yellow]")
    whisper.load_model(model_name)
    console.print(f"[green]✓ Модель '{model_name}' загружена[/green]")


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
        console.print("[dim]Кэш моделей пуст[/dim]")
        return

    current_model = settings.model

    if model:
        if model == current_model:
            console.print(f"[red]Нельзя удалить активную модель '{model}'[/red]")
            raise typer.Exit(1)

        model_file = cache_dir / f"{model}.pt"
        if not model_file.exists():
            console.print(f"[red]Модель '{model}' не найдена[/red]")
            raise typer.Exit(1)

        size_mb = model_file.stat().st_size / (1024 * 1024)
        model_file.unlink()
        console.print(f"[green]✓ Удалена модель '{model}' ({size_mb:.0f} MB)[/green]")
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
            console.print(f"[dim]Удалена: {model_file.name} ({size_mb:.0f} MB)[/dim]")
            deleted_count += 1
            total_size_mb += size_mb

    if deleted_count == 0:
        console.print("[green]Нет неиспользуемых моделей[/green]")
    else:
        console.print(f"[green]✓ Удалено {deleted_count} моделей ({total_size_mb:.0f} MB)[/green]")


@app.command()
def list_models() -> None:
    """Показать загруженные модели Whisper."""
    cache_dir = _get_whisper_cache_dir()
    if not cache_dir.exists():
        console.print("[dim]Кэш моделей пуст[/dim]")
        return

    current_model = settings.model
    models = list(cache_dir.glob("*.pt"))

    if not models:
        console.print("[dim]Нет загруженных моделей[/dim]")
        return

    console.print("\n[bold]Загруженные модели:[/bold]")
    for model_file in sorted(models):
        name = model_file.stem
        size_mb = model_file.stat().st_size / (1024 * 1024)
        is_active = name == current_model or name.startswith(f"{current_model}-")
        marker = " [green]← активная[/green]" if is_active else ""
        console.print(f"  {name} ({size_mb:.0f} MB){marker}")


def _write_info_plist(path: Path) -> None:
    path.write_text(f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>sheptun</string>
    <key>CFBundleIdentifier</key>
    <string>com.sheptun.menubar</string>
    <key>CFBundleName</key>
    <string>Sheptun</string>
    <key>CFBundleDisplayName</key>
    <string>Sheptun</string>
    <key>CFBundleVersion</key>
    <string>{__version__}</string>
    <key>CFBundleShortVersionString</key>
    <string>{__version__}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Sheptun needs microphone access for voice recognition</string>
</dict>
</plist>
""")


def _write_executable(path: Path, project_dir: Path, venv_dir: Path) -> None:
    import stat

    path.write_text(f"""\
#!/bin/bash
cd "{project_dir}"
source "{venv_dir}/bin/activate"
exec python -m sheptun.menubar
""")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _generate_app_icons(icon_src: Path, iconset_dir: Path, output_icns: Path) -> None:
    import shutil
    import subprocess

    from PIL import Image

    if not icon_src.exists():
        return

    img = Image.open(icon_src).convert("RGBA")
    icon_padding_ratio = 0.70
    sizes = [16, 32, 64, 128, 256, 512]

    for size in sizes:
        _save_icon(img, iconset_dir / f"icon_{size}x{size}.png", size, icon_padding_ratio)
        retina_size = size * 2
        if retina_size <= 1024:
            _save_icon(img, iconset_dir / f"icon_{size}x{size}@2x.png", retina_size, icon_padding_ratio)

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(output_icns)],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(iconset_dir)


def _save_icon(img: "Image.Image", path: Path, size: int, padding_ratio: float) -> None:
    from PIL import Image

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon_size = int(size * padding_ratio)
    offset = (size - icon_size) // 2
    resized = img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
    canvas.paste(resized, (offset, offset), resized)
    canvas.save(path)


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
    import shutil
    import sys

    _kill_menubar_app()
    _preload_whisper_model()

    project_dir = Path(__file__).parent.parent.parent
    venv_dir = Path(sys.executable).parent.parent
    resources_dir = Path(__file__).parent / "resources"

    app_dir = output or settings.app_path
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_app_dir = contents_dir / "Resources"
    iconset_dir = resources_app_dir / "AppIcon.iconset"

    if app_dir.exists():
        shutil.rmtree(app_dir)

    macos_dir.mkdir(parents=True)
    iconset_dir.mkdir(parents=True)

    _write_info_plist(contents_dir / "Info.plist")
    _write_executable(macos_dir / "sheptun", project_dir, venv_dir)
    _generate_app_icons(
        resources_dir / "microphone-idle.png",
        iconset_dir,
        resources_app_dir / "AppIcon.icns",
    )

    console.print(f"[green]✓ Приложение создано: {app_dir}[/green]")
    console.print("[dim]Теперь можно запустить из Launchpad или Finder[/dim]")


if __name__ == "__main__":
    app()
