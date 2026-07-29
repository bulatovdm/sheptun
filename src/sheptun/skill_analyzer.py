"""Bridge between the `analyze-replacements-skill` skill and the analyzer pipeline.

The SDK pipeline (`log_analyzer.py`) calls the model itself over HTTP. The skill
instead lets Claude Code subagents be the model, so only the deterministic halves
are needed here:

    plan   -> build context windows, slice them by the checkpoint, emit batches
              as JSON (prompts included, so a subagent needs no extra context)
    commit -> take a subagent's raw JSON reply, run it through the SAME
              normalisation/hallucination guards, append to the report and to
              replacements.yaml, then advance the checkpoint

Everything else — window building, prompts, dedup, YAML writing — is reused from
`log_analyzer` so both entry points stay in sync.
"""

from __future__ import annotations

import fcntl
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sheptun.config import get_config_path, get_replacements_path
from sheptun.log_analyzer import (
    REPLACEMENTS_PROMPT_NAME,
    USER_INTRO_PROMPT_NAME,
    AnalyzerConfig,
    AnalyzerState,
    ContextWindow,
    ContextWindowBuilder,
    LogParser,
    PhraseIndex,
    ReplacementSuggestion,
    SuggestionWriter,
    WindowBatcher,
    _confidence_rank,
    _extract_items,
    _normalize_item,
)
from sheptun.prompts import load_prompt
from sheptun.settings import settings

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Checkpoint kept apart from the SDK pipeline's, so the two never fight over one position.
SKILL_STATE_FILE = "skill_analyzer_state.json"


def skill_state() -> AnalyzerState:
    return AnalyzerState(settings.dataset_path / SKILL_STATE_FILE)


@dataclass(frozen=True)
class BatchPlan:
    """One batch handed to one subagent.

    ``task_path`` holds the complete, ready-to-read task (system prompt + windows).
    The orchestrator passes only this path to the subagent, so the bulky log text
    never has to travel through the expensive orchestrator's context.
    """

    index: int  # 1-based, for display
    start: int  # absolute position of the first window in the full set
    end: int  # absolute position after the last window (the checkpoint if committed)
    task_path: Path


def _render_batch_task(batch: Sequence[ContextWindow]) -> str:
    """The full subagent task: the SAME prompts the SDK client sends.

    System criteria (`replacements_system.md`) plus the user intro and the windows
    (`replacements_user_intro.md`), concatenated — the `known` block is omitted since
    dedup against existing rules happens on our side at commit time.
    """
    blocks: list[str] = [load_prompt(REPLACEMENTS_PROMPT_NAME), "", "---", ""]
    blocks.append(load_prompt(USER_INTRO_PROMPT_NAME))
    blocks.append("")
    for i, window in enumerate(batch, start=1):
        blocks.append(f"=== Фрагмент {i} (частота: {window.frequency}) ===")
        blocks.append(window.render())
        blocks.append("")
    return "\n".join(blocks)


def plan_batches(
    log_path: Path,
    config: AnalyzerConfig,
    start_offset: int,
    max_batches: int,
    task_dir: Path,
    newest_first: bool = True,
) -> tuple[list[BatchPlan], int]:
    """Build the batches this run should process, writing each task to its own file.

    ``newest_first`` (the default) walks the log backwards, newest windows first:
    recent speech is where fresh ASR errors live, while the early history has already
    been mined by previous runs. The checkpoint then counts windows consumed from the
    END, so a resumed run keeps moving further into the past.

    Returns (batches, full_total).
    """
    entries = LogParser().parse(log_path)
    windows = ContextWindowBuilder(config.context_lines, config.min_freq).build(entries)
    full_total = len(windows)
    if newest_first:
        windows = list(reversed(windows))
    pending = windows[start_offset:]
    if config.max_windows > 0:
        pending = pending[: config.max_windows]
    batches = WindowBatcher(config.batch_size).batch(pending)
    if max_batches > 0:
        batches = batches[:max_batches]

    task_dir.mkdir(parents=True, exist_ok=True)
    plans: list[BatchPlan] = []
    cursor = start_offset
    for i, batch in enumerate(batches, start=1):
        task_path = task_dir / f"skill_batch_{i}.txt"
        task_path.write_text(_render_batch_task(batch), encoding="utf-8")
        plans.append(BatchPlan(index=i, start=cursor, end=cursor + len(batch), task_path=task_path))
        cursor += len(batch)
    return plans, full_total


def _phrase_index(log_path: Path) -> PhraseIndex:
    return PhraseIndex(entry.text for entry in LogParser().parse(log_path))


def normalize_reply(
    raw: str,
    log_path: Path,
    min_confidence: str,
    existing_keys: set[str],
) -> tuple[list[ReplacementSuggestion], list[str]]:
    """Turn a subagent's raw reply into vetted suggestions.

    Applies the same three guards as the SDK path: shape normalisation, the
    hallucination check (``old`` must really occur in a Recognized line, and its
    frequency comes from there), the confidence floor, plus dedup against rules
    already present in replacements.yaml. Returns (accepted, rejection notes).
    """
    index = _phrase_index(log_path)
    threshold = _confidence_rank(min_confidence)
    accepted: list[ReplacementSuggestion] = []
    notes: list[str] = []
    seen = set(existing_keys)

    for item in _extract_items(raw):
        normalized = _normalize_item(item, 1)
        if normalized is None:
            notes.append(f"отброшено (пустое/no-op): {item}")
            continue
        frequency = index.frequency(normalized.old)
        if frequency == 0:
            notes.append(f'отброшено (нет в логе): "{normalized.old}"')
            continue
        if _confidence_rank(normalized.confidence) < threshold:
            notes.append(f'отброшено (conf={normalized.confidence}): "{normalized.old}"')
            continue
        key = normalized.old.lower()
        if key in seen:
            notes.append(f'отброшено (дубль): "{normalized.old}"')
            continue
        seen.add(key)
        accepted.append(
            ReplacementSuggestion(
                old=normalized.old,
                new=normalized.new,
                confidence=normalized.confidence,
                reason=normalized.reason,
                frequency=frequency,
            )
        )
    return accepted, notes


def _existing_keys() -> set[str]:
    from sheptun.commands import CommandConfigLoader

    loaded = CommandConfigLoader.load(get_config_path(), get_replacements_path()).replacements
    return {k.lower() for k in loaded}


def _cmd_plan(argv: list[str]) -> int:
    args = _parse_kv(argv)
    log_path = Path(args.get("log", str(settings.log_file)))
    if not log_path.exists():
        print(json.dumps({"error": f"лог не найден: {log_path}"}, ensure_ascii=False))
        return 1

    config = AnalyzerConfig()
    if "context" in args:
        config.context_lines = int(args["context"])
    if "batch-size" in args:
        config.batch_size = int(args["batch-size"])
    if "min-freq" in args:
        config.min_freq = int(args["min-freq"])

    state = skill_state()
    start = 0 if args.get("full") == "1" else int(args.get("start", state.position()))
    max_batches = int(args.get("max-batches", "0"))

    task_dir = Path(args.get("task-dir", "tmp"))
    # Newest-first by default: fresh speech carries the errors worth fixing. `--oldest-first`
    # restores the chronological walk the SDK pipeline uses.
    newest_first = args.get("oldest-first") != "1"
    plans, full_total = plan_batches(log_path, config, start, max_batches, task_dir, newest_first)
    # Only paths and positions travel back to the orchestrator — the prompt text stays
    # in the files, read directly by the subagent that needs it.
    payload = {
        "log": str(log_path),
        "order": "newest-first" if newest_first else "oldest-first",
        "start_position": start,
        "full_total": full_total,
        "batch_size": config.batch_size,
        "batches": [
            {"index": p.index, "start": p.start, "end": p.end, "task": str(p.task_path)}
            for p in plans
        ],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


@contextmanager
def _write_lock() -> Iterator[None]:
    """Serialise the read-modify-write of replacements.yaml and the checkpoint.

    Subagents commit their own batches concurrently, so without this two of them
    could load the same rule set, each append their own rules, and the second write
    would silently drop the first one's.
    """
    lock_path = settings.dataset_path / "skill_analyzer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _cmd_commit(argv: list[str]) -> int:
    """Vet one subagent reply, write the rules out, and advance the checkpoint.

    Called by the subagent that produced the reply, so writes are serialised by
    ``_write_lock``. The checkpoint only moves forward: batches finish out of order
    under concurrency, and a later batch must never pull the position backwards.
    """
    args = _parse_kv(argv)
    log_path = Path(args.get("log", str(settings.log_file)))
    reply_path = Path(args["reply"])
    position = int(args["position"])

    raw = reply_path.read_text(encoding="utf-8")

    with _write_lock():
        accepted, _ = normalize_reply(
            raw,
            log_path,
            args.get("min-confidence", settings.analyzer_min_confidence),
            _existing_keys(),
        )
        applied = SuggestionWriter().apply(accepted, get_replacements_path())
        state = skill_state()
        if args.get("advance", "1") == "1" and position > state.position():
            state.save(position, _runs(state))
        done, total, added = state.position(), _window_total(args), _bump_added(applied)

    scope = f"{done}/{total}, осталось {max(total - done, 0)}" if total else f"{done}"
    print(f"[окна {scope}] новых правил: +{applied}, всего за прогон: {added}")
    return 0


def _window_total(args: dict[str, str]) -> int:
    """Total window count, passed in by the orchestrator to keep commit cheap.

    Recomputing it here would re-parse the whole log on every batch, so the caller
    supplies it once from the plan; without it the progress line just omits the total.
    """
    raw = args.get("total")
    return int(raw) if raw and raw.isdigit() else 0


_ADDED_FILE = "skill_analyzer_added.json"


def _added() -> int:
    """Rules written by THIS run so far.

    Each batch is its own process, so the counter lives on disk next to the
    checkpoint and resets together with it.
    """
    path = settings.dataset_path / _ADDED_FILE
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("added", 0))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0


def _bump_added(applied: int) -> int:
    total = _added() + applied
    path = settings.dataset_path / _ADDED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"added": total}) + "\n", encoding="utf-8")
    return total


def _runs(state: AnalyzerState) -> int:
    if not state.path.exists():
        return 0
    try:
        data: Any = json.loads(state.path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    value = data.get("runs")
    return int(value) if isinstance(value, int) else 0


def _cmd_status(argv: list[str]) -> int:
    """Checkpoint state. With ``--total N`` prints the one-line progress instead of JSON.

    The orchestrator calls this between groups: a subagent's own stdout never reaches
    the user's terminal, so the progress line has to be printed by the caller.
    """
    args = _parse_kv(argv)
    state = skill_state()
    done = state.position()
    total = _window_total(args)
    if total:
        print(
            f"[окна {done}/{total}, осталось {max(total - done, 0)}] правил за прогон: {_added()}"
        )
        return 0
    print(
        json.dumps(
            {
                "state_file": str(state.path),
                "position": done,
                "runs": _runs(state),
                "added": _added(),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_reset(_argv: list[str]) -> int:
    state = skill_state()
    state.reset()
    (settings.dataset_path / _ADDED_FILE).unlink(missing_ok=True)
    print(json.dumps({"reset": str(state.path)}, ensure_ascii=False))
    return 0


def _parse_kv(argv: list[str]) -> dict[str, str]:
    """Parse `--key value` / `--flag` pairs into a plain dict."""
    args: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            args[key] = argv[i + 1]
            i += 2
            continue
        args[key] = "1"
        i += 1
    return args


_COMMANDS = {
    "plan": _cmd_plan,
    "commit": _cmd_commit,
    "status": _cmd_status,
    "reset": _cmd_reset,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _COMMANDS:
        print(f"usage: skill_analyzer <{'|'.join(_COMMANDS)}> [--key value ...]", file=sys.stderr)
        return 2
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
