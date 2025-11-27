import subprocess
from enum import Enum, auto

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from sheptun.i18n import t


def _send_notification(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


class Status(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    ERROR = auto()


STATUS_STYLES: dict[Status, tuple[str, str]] = {
    Status.IDLE: ("dim", "⏸"),
    Status.LISTENING: ("green bold", "🎤"),
    Status.PROCESSING: ("yellow", "⚙"),
    Status.ERROR: ("red bold", "❌"),
}

STATUS_LABELS: dict[Status, str] = {
    Status.IDLE: "Ожидание",
    Status.LISTENING: "Слушаю...",
    Status.PROCESSING: "Обработка...",
    Status.ERROR: "Ошибка",
}


class ConsoleStatusIndicator:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._status = Status.IDLE
        self._message: str = ""
        self._live: Live | None = None

    def start(self) -> None:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def listening(self) -> None:
        self._update_status(Status.LISTENING)

    def processing(self) -> None:
        self._update_status(Status.PROCESSING)

    def error(self, message: str) -> None:
        self._message = message
        self._update_status(Status.ERROR)

    def idle(self) -> None:
        self._message = ""
        self._update_status(Status.IDLE)

    def show_recognized(self, text: str) -> None:
        self._console.print(f"[cyan]Распознано:[/cyan] {text}")

    def show_action(self, action_description: str) -> None:
        self._console.print(f"[green]Действие:[/green] {action_description}")

    def show_help(self) -> None:
        self._console.print("\n[bold cyan]Доступные команды:[/bold cyan]")
        self._console.print(t("help_commands"))
        self._console.print()
        _send_notification(t("help_title"), "Справка выведена в консоль")

    def _update_status(self, status: Status) -> None:
        self._status = status
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Panel:
        style, icon = STATUS_STYLES[self._status]
        label = STATUS_LABELS[self._status]

        content = Text()
        content.append(f"{icon} ", style=style)
        content.append(label, style=style)

        if self._message:
            content.append(f"\n{self._message}", style="dim")

        return Panel(content, title="Sheptun", border_style=style)


class SimpleStatusIndicator:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def start(self) -> None:
        self._console.print("[bold green]Sheptun запущен[/bold green]")

    def stop(self) -> None:
        self._console.print("[bold red]Sheptun остановлен[/bold red]")

    def listening(self) -> None:
        self._console.print("[green]🎤 Слушаю...[/green]")

    def processing(self) -> None:
        self._console.print("[yellow]⚙ Обработка...[/yellow]")

    def error(self, message: str) -> None:
        self._console.print(f"[red]❌ Ошибка: {message}[/red]")

    def idle(self) -> None:
        pass

    def show_recognized(self, text: str) -> None:
        self._console.print(f"[cyan]Распознано:[/cyan] {text}")

    def show_action(self, action_description: str) -> None:
        self._console.print(f"[green]Действие:[/green] {action_description}")

    def show_help(self) -> None:
        self._console.print("\n[bold cyan]Доступные команды:[/bold cyan]")
        self._console.print(t("help_commands"))
        self._console.print()
        _send_notification(t("help_title"), "Справка выведена в консоль")
