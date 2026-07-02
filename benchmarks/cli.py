"""CLI for correction benchmarks: `python -m benchmarks run --correctors sage,jamspell`."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from sheptun.config import get_replacements_path
from sheptun.settings import settings

from . import correctors as corrector_registry
from . import samples as sample_source
from .runner import run
from .types import BenchmarkReport, CorrectorReport, Sample

app = typer.Typer(
    name="benchmarks", help="Бенчмарки корректоров текста для Sheptun", no_args_is_help=True
)
console = Console()


def _load_samples(source: str, count: int, log: Path | None, dedup: bool) -> list[Sample]:
    if source == "log":
        log_path = log or settings.log_file
        if not log_path.exists():
            console.print(f"[red]Лог не найден: {log_path}[/red]")
            raise typer.Exit(1)
        return sample_source.from_log(log_path, count, dedup=dedup)
    if source == "replacements":
        limit = None if count <= 0 else count
        return sample_source.from_replacements(get_replacements_path(), limit)
    console.print(f"[red]Неизвестный источник выборки: {source} (log | replacements)[/red]")
    raise typer.Exit(1)


@contextmanager
def _progress() -> Iterator[Callable[[str, int, int], None]]:
    """A Rich progress bar exposed as an on_progress(name, done, total) callback.

    One task per corrector, created lazily on first tick so ETA is per-corrector.
    """
    bar = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    tasks: dict[str, TaskID] = {}
    with bar:

        def advance(name: str, done: int, total: int) -> None:
            if name not in tasks:
                tasks[name] = bar.add_task(name, total=total)
            bar.update(tasks[name], completed=done)

        yield advance


def _render(report: BenchmarkReport) -> None:
    console.print(
        f"\n[bold]Выборка:[/bold] {report.sample_count} строк · "
        f"с латиницей {report.with_latin} · с англотерминами {report.with_terms} · "
        f"с эталоном {report.with_reference}\n"
    )

    table = Table(title="Результаты корректоров")
    table.add_column("Корректор", style="cyan")
    table.add_column("Изменено", justify="right")
    table.add_column("Повреждено", justify="right", style="red")
    table.add_column("Потеряно lat/термы", justify="right", style="red")
    table.add_column("Точность", justify="right", style="green")
    table.add_column("мс/строку", justify="right")
    for rep in report.correctors:
        acc = f"{rep.exact_match * 100:.0f}%" if rep.exact_match is not None else "—"
        table.add_row(
            rep.name,
            f"{rep.changed} ({rep.changed_pct:.0f}%)",
            f"{rep.damaged} ({rep.damaged_pct:.0f}%)",
            f"{rep.lost_latin}/{rep.lost_terms}",
            acc,
            f"{rep.ms_per_line:.0f}",
        )
    console.print(table)

    for rep in report.correctors:
        _render_examples(rep)


def _render_examples(rep: CorrectorReport) -> None:
    if not rep.examples:
        return
    console.print(f"\n[bold red]Повреждения — {rep.name}:[/bold red]")
    for ex in rep.examples:
        terms = ", ".join(sorted(ex.lost_latin | ex.lost_terms))
        console.print(f"  [red]✗ {terms}[/red]")
        console.print(f"    было : [dim]{ex.sample.text[:100]}[/dim]")
        console.print(f"    стало: {ex.output[:100]}")


@app.command("run")
def run_bench(
    correctors: Annotated[
        str,
        typer.Option("--correctors", help="Список через запятую: noop,jamspell,sage"),
    ] = "noop,sage",
    source: Annotated[
        str,
        typer.Option("--source", help="Источник выборки: log | replacements"),
    ] = "log",
    count: Annotated[
        int,
        typer.Option("--count", help="Размер выборки (0 = весь лог)"),
    ] = 0,
    dedup: Annotated[
        bool,
        typer.Option(
            "--dedup/--no-dedup", help="Только уникальные фразы (быстрее, репрезентативно)"
        ),
    ] = True,
    log: Annotated[
        Path | None,
        typer.Option("--log", help="Путь к логу (по умолчанию из настроек)"),
    ] = None,
) -> None:
    """Прогнать корректоры по выборке и показать урон/точность/скорость."""
    names = [n.strip() for n in correctors.split(",") if n.strip()]
    unknown = [n for n in names if n not in corrector_registry.available()]
    if unknown:
        console.print(
            f"[red]Неизвестные корректоры: {unknown}. "
            f"Доступны: {', '.join(corrector_registry.available())}[/red]"
        )
        raise typer.Exit(1)

    samples = _load_samples(source, count, log, dedup)
    if not samples:
        console.print("[yellow]Пустая выборка.[/yellow]")
        raise typer.Exit(1)
    console.print(f"[dim]Строк к обработке: {len(samples)} × корректоров: {len(names)}[/dim]")

    instances = [corrector_registry.create(n) for n in names]
    with _progress() as advance:
        report = run(samples, instances, on_progress=advance)
    _render(report)


@app.command("list")
def list_correctors() -> None:
    """Показать доступные корректоры."""
    console.print("Доступные корректоры: " + ", ".join(corrector_registry.available()))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
