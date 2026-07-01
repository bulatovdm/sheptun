"""Log analysis for replacement suggestions via the Anthropic Agent SDK.

Pipeline (each stage is a separate, independently configurable component):

    LogParser            -> Recognized lines only (noise dropped)
    ContextWindowBuilder -> per-target windows with +/-N neighbouring lines
    WindowBatcher        -> groups of windows for one API request
    AnthropicClient      -> LLM call, returns structured replacement suggestions
    ReplacementAnalyzer  -> orchestrates, dedups against existing rules
    SuggestionWriter     -> report file and/or apply to replacements.yaml
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from sheptun.prompts import load_prompt
from sheptun.settings import settings

logger = logging.getLogger("sheptun.analyzer")

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_RECOGNIZED_PATTERN = re.compile(r"Recognized: '(?P<text>.*)'\s*$")
_TIMESTAMP_PATTERN = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
REPLACEMENTS_PROMPT_NAME = "replacements_system"
USER_INTRO_PROMPT_NAME = "replacements_user_intro"
VERIFY_PROMPT_NAME = "replacements_verify"

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_TIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$")


def normalize_since(value: str) -> str:
    """Normalize a user date to the log timestamp format for a lower bound.

    A bare date stays as-is: string comparison `"2026-07-01 10:31" > "2026-07-01"`
    already means "that whole day and later".
    """
    return _validate_date(value)


def normalize_until(value: str) -> str:
    """Normalize a user date for an inclusive upper bound.

    A bare date is expanded to end-of-day so the whole day is included.
    """
    normalized = _validate_date(value)
    if _DATE_ONLY.match(normalized):
        return f"{normalized} 23:59:59"
    if len(normalized) == len("YYYY-MM-DD HH:MM"):
        return f"{normalized}:59"
    return normalized


def _validate_date(value: str) -> str:
    value = value.strip()
    if not (_DATE_ONLY.match(value) or _DATE_TIME.match(value)):
        raise ValueError(f"Неверный формат даты: {value!r} (ожидается YYYY-MM-DD[ HH:MM[:SS]])")
    return value


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _confidence_rank(value: str) -> int:
    return _CONFIDENCE_RANK.get(value.lower(), 0)


@dataclass(frozen=True)
class LogEntry:
    """A parsed Recognized line from the log."""

    timestamp: str
    text: str


class PhraseIndex:
    """Case-insensitive frequency lookup over the actual Recognized lines.

    Backs two guards against the model inventing rules: a suggested ``old`` that
    appears in NO Recognized line is a hallucination and must be dropped; one that
    does appear gets its real occurrence count (not a batch-wide maximum).

    ``frequency`` counts Recognized lines that *contain* the phrase as a substring,
    so an inflected/shorter ``old`` (e.g. "комит" inside "сделай комит") still
    matches the line it came from.
    """

    def __init__(self, texts: Iterable[str]) -> None:
        self._lines = tuple(t.lower() for t in texts)

    def frequency(self, phrase: str) -> int:
        needle = phrase.lower().strip()
        if not needle:
            return 0
        return sum(1 for line in self._lines if needle in line)


@dataclass(frozen=True)
class ContextWindow:
    """A target Recognized line plus its surrounding context."""

    target: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    frequency: int = 1
    timestamp: str = ""

    def render(self) -> str:
        lines: list[str] = []
        for entry in self.before:
            lines.append(f"  {entry}")
        lines.append(f">>> {self.target}")
        for entry in self.after:
            lines.append(f"  {entry}")
        return "\n".join(lines)

    def lines(self) -> tuple[str, ...]:
        """Every Recognized line shown to the model for this window."""
        return (*self.before, self.target, *self.after)


@dataclass(frozen=True)
class ReplacementSuggestion:
    old: str
    new: str
    confidence: str
    reason: str
    frequency: int = 1


@dataclass(frozen=True)
class AnalysisResult:
    """Outcome of one run: suggestions plus the windows actually processed.

    ``checkpoint`` is the newest timestamp among processed windows — the value to
    persist so the next run resumes after it.
    """

    suggestions: list[ReplacementSuggestion]
    processed: int
    total: int
    checkpoint: str
    # True when a batch raised (e.g. proxy 502) and the run stopped early. The
    # result still holds the progress from the batches that DID complete, so the
    # caller persists the checkpoint instead of losing everything to a traceback.
    interrupted: bool = False


@dataclass(frozen=True)
class BatchProgress:
    """Progress after one batch (model request) completes.

    ``checkpoint`` is the newest timestamp among windows processed so far — the
    caller can persist it after each batch so an interrupted run still resumes.
    """

    batch_index: int
    batch_total: int
    windows_done: int
    suggestions_so_far: int
    new_suggestions: list[ReplacementSuggestion]
    checkpoint: str


ProgressCallback = Callable[[BatchProgress], None]


@dataclass
class AnalyzerConfig:
    """All tunables for one analysis run. Defaults come from settings."""

    context_lines: int = field(default_factory=lambda: settings.analyzer_context_lines)
    batch_size: int = field(default_factory=lambda: settings.analyzer_batch_size)
    max_windows: int = field(default_factory=lambda: settings.analyzer_max_windows)
    min_freq: int = field(default_factory=lambda: settings.analyzer_min_freq)
    model: str = field(default_factory=lambda: settings.analyzer_model)
    effort: str = field(default_factory=lambda: settings.analyzer_effort)
    min_confidence: str = field(default_factory=lambda: settings.analyzer_min_confidence)
    max_iterations: int = field(default_factory=lambda: settings.analyzer_max_iterations)
    verify: bool = field(default_factory=lambda: settings.analyzer_verify)
    delay: float = field(default_factory=lambda: settings.analyzer_delay)
    retry_backoff: float = field(default_factory=lambda: settings.analyzer_retry_backoff)
    max_error_retries: int = field(default_factory=lambda: settings.analyzer_max_error_retries)
    since: str | None = None  # only target lines with timestamp > since
    until: str | None = None  # only target lines with timestamp <= until
    # Process windows oldest-first so the checkpoint advances contiguously and
    # survives interruption. Auto-forced when max_iterations > 0.
    chronological: bool = False


def _extract_items(text: str) -> list[dict[str, Any]]:
    """Robustly pull a list of suggestion dicts from a possibly-noisy LLM reply.

    Handles markdown code fences and both shapes: a bare JSON array, or an
    object with a "suggestions" array.
    """
    if not text:
        return []

    candidate = _strip_code_fence(text.strip())
    data = _try_json(candidate) or _try_json(_slice_json(candidate))
    if data is None:
        return []

    if isinstance(data, dict):
        data = data.get("suggestions", [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match is not None else text


def _slice_json(text: str) -> str:
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        return text
    end = max(text.rfind("]"), text.rfind("}"))
    return text[start : end + 1] if end > start else text


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(value: str) -> str:
    """Strip control chars and normalise quotes/newlines so the value is safe to
    embed in a double-quoted YAML string and a trailing comment.

    Whisper can produce control chars (e.g. \\x08) and quotes in transcribed text;
    left unescaped they corrupt replacements.yaml so it no longer parses.
    """
    value = _CONTROL_CHARS.sub("", value)
    value = value.replace("\n", " ").replace("\r", " ")
    return value.replace('"', "'").strip()


def _strip_word_boundaries(value: str) -> str:
    """Drop leading/trailing \\b regex anchors a model may add around a word."""
    return re.sub(r"^\\b|\\b$", "", value).strip()


def _normalize_item(item: dict[str, Any], frequency: int) -> ReplacementSuggestion | None:
    old = _sanitize(_strip_word_boundaries(str(item.get("old") or item.get("find") or "").strip()))
    new = _sanitize(
        _strip_word_boundaries(str(item.get("new") or item.get("replace") or "").strip())
    )
    if not old or not new or old.lower() == new.lower():
        return None
    return ReplacementSuggestion(
        old=old,
        new=new,
        confidence=_sanitize(str(item.get("confidence", "medium"))),
        reason=_sanitize(str(item.get("reason", ""))),
        frequency=frequency,
    )


class LogParser:
    """Extracts significant lines from the raw log, drops noise."""

    def parse(self, log_path: Path) -> list[LogEntry]:
        entries: list[LogEntry] = []
        with log_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                entry = self._parse_line(line.rstrip("\n"))
                if entry is not None:
                    entries.append(entry)
        return entries

    def _parse_line(self, line: str) -> LogEntry | None:
        recognized = _RECOGNIZED_PATTERN.search(line)
        if recognized is None:
            return None
        text = recognized.group("text").strip()
        if not text:
            return None
        return LogEntry(timestamp=self._extract_timestamp(line), text=text)

    def _extract_timestamp(self, line: str) -> str:
        match = _TIMESTAMP_PATTERN.match(line)
        return match.group("ts") if match is not None else ""


class ContextWindowBuilder:
    """Builds +/-N context windows around each Recognized line, deduped by frequency.

    ``since``/``until`` (inclusive/inclusive) bound which target lines are kept by
    their timestamp; context lines are always taken from the full log, so a window
    near the boundary still has its surrounding lines. Frequency counts every
    occurrence of a phrase across the whole log, not only the ones in range.
    """

    def __init__(
        self,
        context_lines: int,
        min_freq: int,
        since: str | None = None,
        until: str | None = None,
    ) -> None:
        self._context_lines = context_lines
        self._min_freq = min_freq
        self._since = since
        self._until = until

    def build(self, entries: Sequence[LogEntry]) -> list[ContextWindow]:
        raw_windows = self._build_raw(entries)
        return self._dedup(raw_windows)

    def _build_raw(self, entries: Sequence[LogEntry]) -> list[ContextWindow]:
        windows: list[ContextWindow] = []
        n = self._context_lines
        for index, entry in enumerate(entries):
            before = self._surrounding(entries, start=index - n, end=index)
            after = self._surrounding(entries, start=index + 1, end=index + 1 + n)
            windows.append(
                ContextWindow(
                    target=entry.text, before=before, after=after, timestamp=entry.timestamp
                )
            )
        return windows

    def _surrounding(self, entries: Sequence[LogEntry], start: int, end: int) -> tuple[str, ...]:
        start = max(start, 0)
        return tuple(entries[i].text for i in range(start, min(end, len(entries))))

    def _in_range(self, timestamp: str) -> bool:
        if self._since is not None and timestamp <= self._since:
            return False
        return not (self._until is not None and timestamp > self._until)

    def _dedup(self, windows: Sequence[ContextWindow]) -> list[ContextWindow]:
        counts: dict[str, int] = {}
        first_seen: dict[str, ContextWindow] = {}
        in_range: dict[str, bool] = {}
        # FIRST occurrence timestamp — the window is ordered/checkpointed by it, so a
        # line-wise checkpoint never jumps past unprocessed lines of a frequent phrase.
        first_ts: dict[str, str] = {}
        for window in windows:
            key = window.target.lower()
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, window)
            if window.timestamp:
                current = first_ts.get(key)
                first_ts[key] = (
                    window.timestamp if current is None else min(current, window.timestamp)
                )
            # since/until: keep the phrase if ANY occurrence falls in range (independent
            # of the window's own timestamp above).
            if self._in_range(window.timestamp):
                in_range[key] = True

        result: list[ContextWindow] = []
        for key, window in first_seen.items():
            freq = counts[key]
            if freq < self._min_freq or not in_range.get(key):
                continue
            result.append(
                ContextWindow(
                    target=window.target,
                    before=window.before,
                    after=window.after,
                    frequency=freq,
                    timestamp=first_ts.get(key, window.timestamp),
                )
            )
        result.sort(key=lambda w: w.frequency, reverse=True)
        return result


class WindowBatcher:
    """Packs windows into batches of a given size."""

    def __init__(self, batch_size: int) -> None:
        self._batch_size = max(batch_size, 1)

    def batch(self, windows: Sequence[ContextWindow]) -> list[list[ContextWindow]]:
        return [
            list(windows[i : i + self._batch_size])
            for i in range(0, len(windows), self._batch_size)
        ]


class SuggestClient(Protocol):
    """Anything that turns a batch of windows into replacement suggestions.

    ``known`` are the ``old`` keys already covered (existing rules + earlier
    accepts this run) — the client tells the model not to re-propose them.
    """

    def suggest(
        self, batch: Sequence[ContextWindow], known: set[str] | None = None
    ) -> list[ReplacementSuggestion]: ...

    def set_phrase_index(self, index: PhraseIndex) -> None:
        """Wire the log frequency index (built once the log is parsed)."""
        ...


class AnthropicClient:
    """Thin wrapper over the Anthropic SDK returning structured suggestions."""

    def __init__(
        self,
        model: str,
        effort: str,
        verify: bool = False,
        phrase_index: PhraseIndex | None = None,
    ) -> None:
        self._model = model
        self._effort = effort
        self._verify = verify
        self._phrase_index = phrase_index
        self._client = self._make_client()

    @staticmethod
    def _make_client() -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Пакет 'anthropic' не установлен. Установите: pip install -e '.[llm]'"
            ) from exc

        user_agent = settings.analyzer_user_agent or _DEFAULT_USER_AGENT
        return anthropic.Anthropic(
            base_url=settings.anthropic_base_url,
            api_key=settings.anthropic_api_key,
            default_headers={"User-Agent": user_agent},
            # Fail fast: no SDK backoff/retries, so proxy errors (502 etc.) surface at once
            # instead of stalling the run through several retry waits.
            max_retries=settings.analyzer_max_retries,
        )

    def suggest(
        self, batch: Sequence[ContextWindow], known: set[str] | None = None
    ) -> list[ReplacementSuggestion]:
        text = self._ask(
            load_prompt(REPLACEMENTS_PROMPT_NAME), self._build_prompt(batch, known or set())
        )
        # `old` must occur in the lines actually shown to the model — this is the
        # phrase we replace, so the model cannot invent a form that was never here.
        shown = "\n".join(line for window in batch for line in window.lines()).lower()
        # Fallback only when no index is wired: the batch-wide max is a coarse
        # over-estimate, but it keeps stand-alone client usage working.
        batch_freq = max((w.frequency for w in batch), default=1)
        suggestions = [
            s
            for item in _extract_items(text)
            if (s := self._resolve(_normalize_item(item, batch_freq), shown)) is not None
        ]
        if self._verify and suggestions:
            suggestions = self._verify_suggestions(suggestions)
        return suggestions

    def _resolve(
        self, suggestion: ReplacementSuggestion | None, shown: str
    ) -> ReplacementSuggestion | None:
        """Drop hallucinated ``old``s and attach the real per-word frequency.

        ``shown`` is the lower-cased text of every line handed to the model this
        batch: an ``old`` absent from it was invented, not observed, and is
        dropped. Frequency then comes from the whole-log index (true reach); with
        no index the batch max is kept as a coarse fallback.
        """
        if suggestion is None:
            return None
        if suggestion.old.lower() not in shown:
            return None
        if self._phrase_index is None:
            return suggestion
        return replace(suggestion, frequency=self._phrase_index.frequency(suggestion.old))

    def set_phrase_index(self, index: PhraseIndex) -> None:
        self._phrase_index = index

    def _ask(self, system: str, user: str) -> str:
        """One model call; returns the text of the first text block."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self._effort},
            messages=[{"role": "user", "content": user}],
        )
        return next(
            (block.text for block in response.content if getattr(block, "type", "") == "text"),
            "",
        )

    def _verify_suggestions(
        self, suggestions: list[ReplacementSuggestion]
    ) -> list[ReplacementSuggestion]:
        """Second pass: a critic prompt rejects unsafe/dubious candidates."""
        listing = "\n".join(f'{i}. "{s.old}" -> "{s.new}"' for i, s in enumerate(suggestions, 1))
        text = self._ask(load_prompt(VERIFY_PROMPT_NAME), f"Проверь эти правила:\n{listing}")
        rejected = {
            (str(v.get("old", "")).lower(), str(v.get("new", "")).lower())
            for v in _extract_items(text)
            if str(v.get("verdict", "")).lower() == "reject"
        }
        return [s for s in suggestions if (s.old.lower(), s.new.lower()) not in rejected]

    def _build_prompt(self, batch: Sequence[ContextWindow], known: set[str]) -> str:
        blocks: list[str] = [load_prompt(USER_INTRO_PROMPT_NAME), ""]
        if known:
            blocks.append(
                "УЖЕ ПОКРЫТЫЕ ОШИБКИ (эти слова и их близкие вариации НЕ предлагай — "
                "они уже исправляются существующими правилами):"
            )
            blocks.append(", ".join(sorted(known)))
            blocks.append("")
        for i, window in enumerate(batch, start=1):
            blocks.append(f"=== Фрагмент {i} (частота: {window.frequency}) ===")
            blocks.append(window.render())
            blocks.append("")
        return "\n".join(blocks)


class NoopClient:
    """Client stub for dry runs — never calls the API."""

    def suggest(
        self, batch: Sequence[ContextWindow], known: set[str] | None = None
    ) -> list[ReplacementSuggestion]:
        del batch, known  # dry-run stub — never calls the API
        return []

    def set_phrase_index(self, index: PhraseIndex) -> None:
        del index  # dry-run stub — nothing to index


class ReplacementAnalyzer:
    """Orchestrates the pipeline and dedups against existing rules."""

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        client: SuggestClient | None = None,
    ) -> None:
        self._config = config or AnalyzerConfig()
        self._parser = LogParser()
        self._window_builder = ContextWindowBuilder(
            self._config.context_lines,
            self._config.min_freq,
            since=self._config.since,
            until=self._config.until,
        )
        self._batcher = WindowBatcher(self._config.batch_size)
        self._client = client or AnthropicClient(
            self._config.model, self._config.effort, verify=self._config.verify
        )

    def analyze(self, log_path: Path, existing: dict[str, str] | None = None) -> AnalysisResult:
        windows = self.prepare_windows(log_path)
        return self.analyze_windows(windows, existing or {})

    def prepare_windows(self, log_path: Path) -> list[ContextWindow]:
        entries = self._parser.parse(log_path)
        # Real per-word frequencies over every Recognized line — lets the client
        # reject a hallucinated ``old`` and stamp the true count on the rest.
        self._client.set_phrase_index(PhraseIndex(e.text for e in entries))
        windows = self._window_builder.build(entries)
        # Chronological order lets the checkpoint advance contiguously (and survive
        # interruption). An iteration cap forces it too, since it processes a prefix.
        # Otherwise rank by frequency (most impactful first).
        if self._config.chronological or self._config.max_iterations > 0:
            windows.sort(key=lambda w: w.timestamp)
        if self._config.max_windows > 0:
            windows = windows[: self._config.max_windows]
        return windows

    def analyze_windows(
        self,
        windows: Sequence[ContextWindow],
        existing: dict[str, str],
        on_progress: ProgressCallback | None = None,
    ) -> AnalysisResult:
        existing_keys = {k.lower() for k in existing}
        batches = self._batcher.batch(windows)
        if self._config.max_iterations > 0:
            batches = batches[: self._config.max_iterations]

        threshold = _confidence_rank(self._config.min_confidence)
        seen: set[str] = set(existing_keys)
        accepted: list[ReplacementSuggestion] = []
        processed: list[ContextWindow] = []

        running_checkpoint = ""
        interrupted = False
        for index, batch in enumerate(batches, start=1):
            # Pause between requests (not before the first) to ease load on the
            # proxy/origin — a gap-less stream of 3600+ calls overloads it (502s).
            if index > 1 and self._config.delay > 0:
                time.sleep(self._config.delay)
            try:
                raw = self._suggest_with_retry(batch, seen, index, len(batches))
            except Exception:
                # Batch still failing after all retries. Stop, but return what the
                # earlier batches produced so the caller can persist the checkpoint —
                # otherwise the whole run's progress is lost to a traceback.
                interrupted = True
                break
            fresh = self._accept_new(raw, threshold, seen)
            accepted.extend(fresh)
            processed.extend(batch)
            running_checkpoint = max(
                running_checkpoint, max((w.timestamp for w in batch), default="")
            )
            if on_progress is not None:
                on_progress(
                    BatchProgress(
                        batch_index=index,
                        batch_total=len(batches),
                        windows_done=len(processed),
                        suggestions_so_far=len(accepted),
                        new_suggestions=fresh,
                        checkpoint=running_checkpoint,
                    )
                )

        return AnalysisResult(
            suggestions=accepted,
            processed=len(processed),
            total=len(windows),
            checkpoint=running_checkpoint,
            interrupted=interrupted,
        )

    def _suggest_with_retry(
        self, batch: Sequence[ContextWindow], seen: set[str], index: int, total: int
    ) -> list[ReplacementSuggestion]:
        """Call the client, retrying a failed request with a growing backoff.

        On error: log it and wait ``retry_backoff * attempt`` seconds (15, 30, 45, …),
        then retry the SAME batch. After ``max_error_retries`` failures give up and
        re-raise the last error so the caller stops the run (keeping earlier progress).
        A success resets the counter. ``KeyboardInterrupt`` is never retried.
        """
        attempt = 0
        while True:
            try:
                return self._client.suggest(batch, known=set(seen))
            except Exception as exc:
                attempt += 1
                if attempt > self._config.max_error_retries:
                    logger.error(
                        "Батч %d/%d — ошибка запроса, попытки исчерпаны (%d): %s",
                        index,
                        total,
                        self._config.max_error_retries,
                        exc,
                    )
                    raise
                wait = self._config.retry_backoff * attempt
                logger.warning(
                    "Батч %d/%d — ошибка запроса (попытка %d/%d): %s. Жду %.0fс и повторяю.",
                    index,
                    total,
                    attempt,
                    self._config.max_error_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

    def _accept_new(
        self,
        suggestions: Iterable[ReplacementSuggestion],
        threshold: int,
        seen: set[str],
    ) -> list[ReplacementSuggestion]:
        """Filter a batch by confidence and dedup against everything seen so far.

        Mutates ``seen`` so the same rule is never emitted twice across batches.
        """
        fresh: list[ReplacementSuggestion] = []
        for suggestion in suggestions:
            if _confidence_rank(suggestion.confidence) < threshold:
                continue
            key = suggestion.old.lower()
            if key in seen:
                continue
            seen.add(key)
            fresh.append(suggestion)
        return fresh


def _report_line(s: ReplacementSuggestion) -> str:
    # Sanitize again at write time — a suggestion may be built directly, not only
    # via _normalize_item, and control chars/quotes must never reach the YAML file.
    old, new = _sanitize(s.old), _sanitize(s.new)
    conf, reason = _sanitize(s.confidence), _sanitize(s.reason)
    return f'"{old}": "{new}"  # freq={s.frequency}, conf={conf} — {reason}'


class SuggestionWriter:
    """Writes suggestions to a report file or appends to replacements.yaml."""

    def write_report(self, suggestions: Sequence[ReplacementSuggestion], output_path: Path) -> int:
        return self.append_report(suggestions, output_path)

    def append_report(self, suggestions: Sequence[ReplacementSuggestion], output_path: Path) -> int:
        """Append suggestions to the report — called after each batch.

        The file is created lazily on the first non-empty write, so a run that
        finds nothing leaves no empty report behind.
        """
        if not suggestions:
            return 0
        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "# Suggested replacements (review before merging into replacements.yaml)\n"
                "# Generated by `sheptun analyze-replacements`\n\n",
                encoding="utf-8",
            )
        with output_path.open("a", encoding="utf-8") as f:
            for s in suggestions:
                f.write(_report_line(s) + "\n")
        return len(suggestions)

    def apply(self, suggestions: Sequence[ReplacementSuggestion], replacements_path: Path) -> int:
        existing_keys = {k.lower() for k in self._load(replacements_path)}
        fresh = [s for s in suggestions if s.old.lower() not in existing_keys]
        if not fresh:
            return 0
        # Append new rules (with reason comments) instead of rewriting the whole
        # file — keeps existing comments and works incrementally per batch.
        replacements_path.parent.mkdir(parents=True, exist_ok=True)
        needs_nl = replacements_path.exists() and not replacements_path.read_text(
            encoding="utf-8"
        ).endswith("\n")
        with replacements_path.open("a", encoding="utf-8") as f:
            if needs_nl:
                f.write("\n")
            for s in fresh:
                f.write(_report_line(s) + "\n")
        return len(fresh)

    def _load(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(loaded) if isinstance(loaded, dict) else {}


class AnalyzerState:
    """Persists the analysis checkpoint (last processed timestamp) as JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (settings.dataset_path / "analyzer_state.json")

    @property
    def path(self) -> Path:
        return self._path

    def last_timestamp(self) -> str | None:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        value = data.get("last_timestamp")
        return str(value) if value else None

    def save(self, last_timestamp: str, runs: int | None = None) -> None:
        if not last_timestamp:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_timestamp": last_timestamp, "runs": (runs or 0) + 1}
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def reset(self) -> None:
        self._path.unlink(missing_ok=True)
