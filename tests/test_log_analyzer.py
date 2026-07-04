import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from sheptun.log_analyzer import (
    AnalyzerConfig,
    AnalyzerState,
    AnthropicClient,
    BatchProgress,
    ContextWindow,
    ContextWindowBuilder,
    LogParser,
    PhraseIndex,
    ReplacementAnalyzer,
    ReplacementSuggestion,
    RetryEvent,
    SuggestionWriter,
    WindowBatcher,
    _extract_items,
    _normalize_item,
    normalize_since,
    normalize_until,
)

SAMPLE_LOG = """\
2026-07-01 10:31:08,035 [INFO] Buffer reset
2026-07-01 10:31:08,100 [INFO] Speech detected
2026-07-01 10:31:15,491 [INFO] Recognized: 'сделай комит'
2026-07-01 10:31:20,000 [INFO] Window changed: A:file.py -> B:other.py
2026-07-01 10:31:22,238 [INFO] Recognized: 'запусти пайтон скрипт'
2026-07-01 10:31:25,000 [INFO] Warmup completed
2026-07-01 10:31:26,000 [INFO] Recognized: 'сделай комит'
2026-07-02 09:00:00,000 [INFO] Recognized: 'открой докер'
"""


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    path = tmp_path / "sheptun.log"
    path.write_text(SAMPLE_LOG, encoding="utf-8")
    return path


class TestLogParser:
    def test_extracts_only_recognized_lines(self, log_path: Path) -> None:
        entries = LogParser().parse(log_path)
        texts = [e.text for e in entries]
        # 4 Recognized lines; noise and Window-changed dropped
        assert texts == [
            "сделай комит",
            "запусти пайтон скрипт",
            "сделай комит",
            "открой докер",
        ]

    def test_drops_noise_and_window_lines(self, log_path: Path) -> None:
        entries = LogParser().parse(log_path)
        texts = [e.text for e in entries]
        assert "Buffer reset" not in texts
        assert not any("Window changed" in t or "[app]" in t for t in texts)

    def test_parses_timestamp(self, log_path: Path) -> None:
        entries = LogParser().parse(log_path)
        assert entries[0].timestamp == "2026-07-01 10:31:15"


class TestContextWindowBuilder:
    def test_window_includes_neighbour_lines(self, log_path: Path) -> None:
        entries = LogParser().parse(log_path)
        windows = ContextWindowBuilder(context_lines=1, min_freq=1).build(entries)
        first = next(w for w in windows if w.target == "запусти пайтон скрипт")
        rendered = first.render()
        assert ">>> запусти пайтон скрипт" in rendered
        assert "сделай комит" in rendered  # neighbouring Recognized line as context
        assert "[app]" not in rendered  # window-change lines are no longer included

    def test_dedup_and_frequency(self, log_path: Path) -> None:
        entries = LogParser().parse(log_path)
        windows = ContextWindowBuilder(context_lines=2, min_freq=1).build(entries)
        komit = next(w for w in windows if w.target == "сделай комит")
        assert komit.frequency == 2

    def test_min_freq_filters(self, log_path: Path) -> None:
        entries = LogParser().parse(log_path)
        windows = ContextWindowBuilder(context_lines=2, min_freq=2).build(entries)
        targets = {w.target for w in windows}
        assert targets == {"сделай комит"}


class TestWindowBatcher:
    def test_splits_into_batches(self) -> None:
        windows = [ContextWindow(target=str(i), before=(), after=()) for i in range(5)]
        batches = WindowBatcher(batch_size=2).batch(windows)
        assert [len(b) for b in batches] == [2, 2, 1]


class _FakeClient:
    def __init__(self, suggestions: list[ReplacementSuggestion]) -> None:
        self._suggestions = suggestions
        self.calls = 0
        self.index: PhraseIndex | None = None

    def suggest(
        self,
        batch: Sequence[ContextWindow],  # noqa: ARG002
        known: set[str] | None = None,  # noqa: ARG002
    ) -> list[ReplacementSuggestion]:
        self.calls += 1
        return self._suggestions

    def set_phrase_index(self, index: PhraseIndex) -> None:
        self.index = index


class _KnownCapturingClient:
    """Fake client that records the union of every `known` set it received."""

    def __init__(self, suggestions: list[ReplacementSuggestion]) -> None:
        self._suggestions = suggestions
        self.seen_known: set[str] = set()

    def suggest(
        self,
        batch: Sequence[ContextWindow],  # noqa: ARG002
        known: set[str] | None = None,
    ) -> list[ReplacementSuggestion]:
        self.seen_known |= known or set()
        return self._suggestions

    def set_phrase_index(self, index: PhraseIndex) -> None:
        del index


class TestReplacementAnalyzer:
    def test_dedups_against_existing(self, log_path: Path) -> None:
        client = _FakeClient(
            [
                ReplacementSuggestion("комит", "коммит", "high", "", 2),
                ReplacementSuggestion("пайтон", "Python", "high", "", 1),
            ]
        )
        config = AnalyzerConfig(context_lines=2, min_freq=1, batch_size=10)
        analyzer = ReplacementAnalyzer(config, client=client)
        result = analyzer.analyze(log_path, existing={"комит": "коммит"})
        olds = {s.old for s in result.suggestions}
        assert olds == {"пайтон"}  # existing rule filtered out

    def test_send_known_off_keeps_prompt_clean_but_still_dedups(self, log_path: Path) -> None:
        # Off by default: the model is never told about existing rules (lean prompt),
        # yet the existing `old` is still filtered out before the result — dedup lives
        # in _accept_new, not in the prompt.
        client = _KnownCapturingClient(
            [
                ReplacementSuggestion("комит", "коммит", "high", "", 2),
                ReplacementSuggestion("пайтон", "Python", "high", "", 1),
            ]
        )
        config = AnalyzerConfig(context_lines=2, min_freq=1, batch_size=10, send_known=False)
        analyzer = ReplacementAnalyzer(config, client=client)
        result = analyzer.analyze(log_path, existing={"комит": "коммит"})
        assert client.seen_known == set()  # nothing about existing rules reached the model
        assert {s.old for s in result.suggestions} == {"пайтон"}  # existing rule still dropped

    def test_send_known_on_passes_existing_to_model(self, log_path: Path) -> None:
        client = _KnownCapturingClient([])
        config = AnalyzerConfig(context_lines=2, min_freq=1, batch_size=10, send_known=True)
        analyzer = ReplacementAnalyzer(config, client=client)
        analyzer.analyze(log_path, existing={"комит": "коммит"})
        assert "комит" in client.seen_known  # opt-in restores the old behaviour

    def test_filters_below_min_confidence(self, log_path: Path) -> None:
        client = _FakeClient(
            [
                ReplacementSuggestion("страку", "строку", "high", "", 3),
                ReplacementSuggestion("пайст", "paste", "low", "", 2),
            ]
        )
        config = AnalyzerConfig(context_lines=2, min_freq=1, batch_size=10, min_confidence="medium")
        analyzer = ReplacementAnalyzer(config, client=client)
        result = analyzer.analyze(log_path, existing={})
        olds = {s.old for s in result.suggestions}
        assert olds == {"страку"}  # low-confidence dropped

    def test_max_windows_limits(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=10, max_windows=1)
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)
        assert len(windows) == 1


class TestTimeRange:
    def test_since_rewinds_offset_to_date(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(
            context_lines=1, min_freq=1, batch_size=10, since="2026-07-01 12:00:00"
        )
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)
        # since is a positional rewind: the set starts at the first window whose
        # first-occurrence timestamp is >= the date. Everything before it is skipped.
        assert all(w.timestamp >= "2026-07-01 12:00:00" for w in windows)
        assert "открой докер" in {w.target for w in windows}

    def test_until_caps_range(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(
            context_lines=1, min_freq=1, batch_size=10, until="2026-07-01 23:59:59"
        )
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)
        targets = {w.target for w in windows}
        assert "открой докер" not in targets  # 2026-07-02 excluded
        assert "сделай комит" in targets

    def _dated_log(self, tmp_path: Path, n: int) -> Path:
        lines = "".join(
            f"2026-01-{i:02d} 10:00:00,000 [INFO] Recognized: 'фраза {i:02d}'\n"
            for i in range(1, n + 1)
        )
        log = tmp_path / "sheptun.log"
        log.write_text(lines, encoding="utf-8")
        return log

    def test_since_position_is_absolute_in_full_set(self, tmp_path: Path) -> None:
        # 10 windows, one per day. --since 2026-01-06 skips the first 5 (positions 0..4),
        # so processing 2 of the remaining must report an ABSOLUTE position of 5+2=7 in
        # the full set — NOT 2 relative to the truncated set. Persisting 2 would make the
        # next incremental run resume at window 2 (early January), re-doing 5 windows.
        log = self._dated_log(tmp_path, n=10)
        config = AnalyzerConfig(
            context_lines=0,
            min_freq=1,
            batch_size=1,
            max_iterations=2,
            since="2026-01-06 00:00:00",
        )
        analyzer = ReplacementAnalyzer(config, client=_FakeClient([]))
        windows = analyzer.prepare_windows(log)
        result = analyzer.analyze_windows(windows, existing={})

        assert result.full_total == 10
        assert result.processed == 2
        assert result.position == 7  # 5 skipped by --since + 2 processed

    def test_until_position_stays_absolute(self, tmp_path: Path) -> None:
        # --until keeps the lower part of the set; processing all of it must reach the
        # position of the last kept window in the FULL set, not restart the count.
        log = self._dated_log(tmp_path, n=10)
        config = AnalyzerConfig(
            context_lines=0, min_freq=1, batch_size=10, until="2026-01-04 23:59:59"
        )
        analyzer = ReplacementAnalyzer(config, client=_FakeClient([]))
        windows = analyzer.prepare_windows(log)
        result = analyzer.analyze_windows(windows, existing={})

        assert result.full_total == 10
        assert result.processed == 4  # фразы 01..04
        assert result.position == 4  # positions 0..3 done → next is index 4


class TestIterationLimitAndCheckpoint:
    def test_max_iterations_limits_batches_and_advances_position(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=1, max_iterations=1)
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)
        result = analyzer.analyze_windows(windows, existing={})
        assert result.processed == 1
        assert result.total == len(windows)
        assert client.calls == 1
        # one batch of one window processed → position advanced by one
        assert result.position == 1
        assert result.full_total == len(windows)

    def test_position_reaches_full_total_when_all_processed(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=10)
        analyzer = ReplacementAnalyzer(config, client=client)
        result = analyzer.analyze(log_path, existing={})
        assert result.position == result.full_total


class TestProgress:
    def test_callback_fires_per_batch_with_fresh_only(self, log_path: Path) -> None:
        client = _FakeClient([ReplacementSuggestion("комит", "коммит", "high", "", 2)])
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=1, concurrency=1)
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)

        seen_batches: list[int] = []
        fresh_counts: list[int] = []

        def on_progress(p: BatchProgress) -> None:
            seen_batches.append(p.batch_index)
            fresh_counts.append(len(p.new_suggestions))

        analyzer.analyze_windows(windows, existing={}, on_progress=on_progress)
        assert seen_batches == list(range(1, len(windows) + 1))
        # same rule returned every batch, but only the first batch emits it as fresh
        assert fresh_counts[0] == 1
        assert sum(fresh_counts) == 1


class _FailingClient:
    """Succeeds for the first `fail_on - 1` batches, then raises (simulates Ctrl+C)."""

    def __init__(self, fail_on: int) -> None:
        self._fail_on = fail_on
        self.calls = 0

    def suggest(
        self,
        batch: Sequence[ContextWindow],  # noqa: ARG002
        known: set[str] | None = None,  # noqa: ARG002
    ) -> list[ReplacementSuggestion]:
        self.calls += 1
        if self.calls >= self._fail_on:
            raise KeyboardInterrupt("interrupted mid-run")
        return []

    def set_phrase_index(self, index: PhraseIndex) -> None:
        del index


class TestStableOrder:
    """The window set is ordered by each phrase's FIRST occurrence, so appending
    new lines to the log never shifts the already-processed prefix — the positional
    checkpoint stays valid across runs."""

    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        p = tmp_path / "sheptun.log"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_window_timestamp_is_first_occurrence(self, tmp_path: Path) -> None:
        # A frequent phrase occurs at 10:00 and again at 20:00. Ordering by the
        # FIRST occurrence (10:00) keeps the phrase's position stable even when the
        # later occurrence is appended after an earlier run.
        log = self._write(
            tmp_path,
            [
                "2026-01-01 10:00:00,000 [INFO] Recognized: 'частая фраза'",
                "2026-01-01 12:00:00,000 [INFO] Recognized: 'другая фраза'",
                "2026-01-01 20:00:00,000 [INFO] Recognized: 'частая фраза'",
            ],
        )
        entries = LogParser().parse(log)
        windows = ContextWindowBuilder(context_lines=0, min_freq=2).build(entries)
        w = next(x for x in windows if x.target == "частая фраза")
        assert w.timestamp == "2026-01-01 10:00:00", (
            f"window timestamp {w.timestamp} is the last occurrence, not the first — "
            "ordering by it would shift the phrase's position when lines are appended"
        )


class TestCheckpointOnInterrupt:
    """A long run interrupted mid-way must still advance the position for the
    batches already processed, so the next run resumes instead of restarting."""

    def test_progress_exposes_running_position(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(
            context_lines=1, min_freq=1, batch_size=1, max_iterations=10, concurrency=1
        )
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)

        positions: list[int] = []
        analyzer.analyze_windows(
            windows, existing={}, on_progress=lambda p: positions.append(p.position)
        )
        # position grows by one per single-window batch and reaches the full total
        assert positions == list(range(1, len(windows) + 1))
        assert positions[-1] == analyzer.full_total

    def test_position_saved_before_interrupt(self, log_path: Path) -> None:
        client = _FailingClient(fail_on=2)  # 1 batch ok, 2nd raises
        config = AnalyzerConfig(
            context_lines=1, min_freq=1, batch_size=1, max_iterations=10, concurrency=1
        )
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)

        saved: list[int] = []

        def on_progress(p: BatchProgress) -> None:
            saved.append(p.position)  # caller persists after each batch

        with pytest.raises(KeyboardInterrupt):
            analyzer.analyze_windows(windows, existing={}, on_progress=on_progress)

        # the first batch's position was delivered before the crash → not lost
        assert saved == [1]


class _NetFailingClient:
    """Raises a network-style error (proxy 502) on the Nth batch — an Exception,
    not KeyboardInterrupt, so the analyzer must swallow it and return partial work."""

    def __init__(self, fail_on: int) -> None:
        self._fail_on = fail_on
        self.calls = 0

    def suggest(
        self,
        batch: Sequence[ContextWindow],  # noqa: ARG002
        known: set[str] | None = None,  # noqa: ARG002
    ) -> list[ReplacementSuggestion]:
        self.calls += 1
        if self.calls >= self._fail_on:
            raise RuntimeError("Error code: 502 - Bad gateway")
        return []

    def set_phrase_index(self, index: PhraseIndex) -> None:
        del index


class TestCheckpointOnRequestError:
    """Regression: a proxy 502 mid-run must NOT crash the analyzer with a traceback.
    It stops early, returns a partial AnalysisResult, and keeps the position of the
    batches that completed — so the next run resumes instead of restarting from zero."""

    def _windows(self, tmp_path: Path, n: int) -> tuple[AnalyzerConfig, Path]:
        lines = "".join(
            f"2026-01-{i:02d} 10:00:00,000 [INFO] Recognized: 'фраза {i}'\n"
            for i in range(1, n + 1)
        )
        log = tmp_path / "sheptun.log"
        log.write_text(lines, encoding="utf-8")
        # delay/backoff/retries=0 → no real sleeps, no retries: these tests assert the
        # give-up path. Retry behaviour is covered separately in TestRetryBackoff.
        config = AnalyzerConfig(
            context_lines=0,
            min_freq=1,
            batch_size=1,
            max_iterations=10,
            delay=0,
            retry_backoff=0,
            max_error_retries=0,
            concurrency=1,  # these assert the strict sequential interrupt/position path
        )
        return config, log

    def test_error_does_not_propagate_and_keeps_progress(self, tmp_path: Path) -> None:
        config, log = self._windows(tmp_path, n=5)
        analyzer = ReplacementAnalyzer(config, client=_NetFailingClient(fail_on=3))
        windows = analyzer.prepare_windows(log)

        # must NOT raise — returns partial result instead of a traceback
        result = analyzer.analyze_windows(windows, existing={})

        assert result.interrupted is True
        assert result.processed == 2  # batches 1 and 2 completed before the 502
        # position is the count of completed windows — the resume point
        assert result.position == 2

    def test_error_on_first_batch_yields_empty_but_no_crash(self, tmp_path: Path) -> None:
        config, log = self._windows(tmp_path, n=3)
        analyzer = ReplacementAnalyzer(config, client=_NetFailingClient(fail_on=1))
        windows = analyzer.prepare_windows(log)

        result = analyzer.analyze_windows(windows, existing={})

        assert result.interrupted is True
        assert result.processed == 0
        assert result.position == 0  # nothing completed → resume from the start offset

    def test_clean_run_is_not_marked_interrupted(self, tmp_path: Path) -> None:
        config, log = self._windows(tmp_path, n=3)
        analyzer = ReplacementAnalyzer(config, client=_NetFailingClient(fail_on=99))
        windows = analyzer.prepare_windows(log)

        result = analyzer.analyze_windows(windows, existing={})

        assert result.interrupted is False
        assert result.processed == 3
        assert result.position == result.full_total


class _FlakyClient:
    """Fails the first ``fail_times`` suggest() calls, then succeeds — models a
    proxy that returns 502 a few times before the origin recovers."""

    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.calls = 0

    def suggest(
        self,
        batch: Sequence[ContextWindow],  # noqa: ARG002
        known: set[str] | None = None,  # noqa: ARG002
    ) -> list[ReplacementSuggestion]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("Error code: 502 - Bad gateway")
        return []

    def set_phrase_index(self, index: PhraseIndex) -> None:
        del index


class TestRetryBackoff:
    """A failing batch is retried with a growing backoff before giving up."""

    def _analyzer(self, tmp_path: Path, client: object, retries: int) -> ReplacementAnalyzer:
        log = tmp_path / "sheptun.log"
        log.write_text("2026-01-01 10:00:00,000 [INFO] Recognized: 'фраза'\n", encoding="utf-8")
        config = AnalyzerConfig(
            context_lines=0,
            min_freq=1,
            batch_size=1,
            max_iterations=10,
            delay=0,
            retry_backoff=15,
            max_error_retries=retries,
        )
        return ReplacementAnalyzer(config, client=client)  # type: ignore[arg-type]

    def test_recovers_after_transient_errors(self, tmp_path: Path, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("sheptun.log_analyzer.time.sleep", slept.append)
        client = _FlakyClient(fail_times=2)  # fail twice, then succeed
        analyzer = self._analyzer(tmp_path, client, retries=4)
        windows = analyzer.prepare_windows(log_path=tmp_path / "sheptun.log")

        result = analyzer.analyze_windows(windows, existing={})

        assert result.interrupted is False  # recovered, run completed
        assert result.processed == 1
        assert client.calls == 3  # 2 failures + 1 success
        assert slept == [15, 30]  # backoff grew: 15*1, 15*2

    def test_gives_up_after_max_retries(self, tmp_path: Path, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("sheptun.log_analyzer.time.sleep", slept.append)
        client = _FlakyClient(fail_times=99)  # never recovers
        analyzer = self._analyzer(tmp_path, client, retries=4)
        windows = analyzer.prepare_windows(log_path=tmp_path / "sheptun.log")

        result = analyzer.analyze_windows(windows, existing={})

        assert result.interrupted is True  # gave up, but no traceback
        assert client.calls == 5  # 1 initial + 4 retries
        assert slept == [15, 30, 45, 60]  # 4 growing waits, then exit

    def test_on_retry_callback_fires_per_attempt_and_on_give_up(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr("sheptun.log_analyzer.time.sleep", lambda _s: None)
        events: list[RetryEvent] = []
        client = _FlakyClient(fail_times=99)  # never recovers → retries then gives up
        analyzer = self._analyzer(tmp_path, client, retries=2)
        windows = analyzer.prepare_windows(log_path=tmp_path / "sheptun.log")

        analyzer.analyze_windows(windows, existing={}, on_retry=events.append)

        # 2 retry events (with waits) + 1 give-up event
        assert [e.wait for e in events] == [15, 30, 0]
        assert [e.gave_up for e in events] == [False, False, True]
        assert all("502" in e.error for e in events)


class TestIncrementalWriter:
    def test_lazy_header_written_once(self, tmp_path: Path) -> None:
        report = tmp_path / "r.yaml"
        writer = SuggestionWriter()
        writer.append_report([ReplacementSuggestion("комит", "коммит", "high", "x", 5)], report)
        writer.append_report([ReplacementSuggestion("докер", "docker", "high", "y", 3)], report)
        content = report.read_text(encoding="utf-8")
        assert '"комит": "коммит"' in content
        assert '"докер": "docker"' in content
        assert content.count("# Suggested replacements") == 1  # header written once

    def test_no_file_when_nothing_found(self, tmp_path: Path) -> None:
        report = tmp_path / "r.yaml"
        SuggestionWriter().append_report([], report)
        assert not report.exists()  # empty run leaves no report behind


class TestAnalyzerState:
    def test_roundtrip(self, tmp_path: Path) -> None:
        state = AnalyzerState(tmp_path / "state.json")
        assert state.position() == 0
        state.save(42)
        assert state.position() == 42

    def test_save_can_rewind(self, tmp_path: Path) -> None:
        # --since/--full rewind the checkpoint, so save must accept a smaller value.
        state = AnalyzerState(tmp_path / "state.json")
        state.save(500)
        state.save(0)
        assert state.position() == 0

    def test_pre_position_file_reads_as_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text('{"last_timestamp": "2026-04-08 16:24:53"}', encoding="utf-8")
        assert AnalyzerState(path).position() == 0

    def test_reset(self, tmp_path: Path) -> None:
        state = AnalyzerState(tmp_path / "state.json")
        state.save(10)
        state.reset()
        assert state.position() == 0

    def test_ignores_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("{ not json", encoding="utf-8")
        assert AnalyzerState(path).position() == 0


class TestDateNormalization:
    def test_since_bare_date_unchanged(self) -> None:
        assert normalize_since("2026-07-01") == "2026-07-01"

    def test_until_bare_date_to_end_of_day(self) -> None:
        assert normalize_until("2026-07-01") == "2026-07-01 23:59:59"

    def test_until_minute_precision_filled(self) -> None:
        assert normalize_until("2026-07-01 10:30") == "2026-07-01 10:30:59"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="формат даты"):
            normalize_since("01.07.2026")


class TestResponseParsing:
    def test_extracts_from_markdown_fence(self) -> None:
        text = '```json\n[{"old": "комит", "new": "коммит"}]\n```'
        items = _extract_items(text)
        assert items == [{"old": "комит", "new": "коммит"}]

    def test_extracts_object_with_suggestions(self) -> None:
        text = '{"suggestions": [{"old": "докер", "new": "docker"}]}'
        items = _extract_items(text)
        assert items == [{"old": "докер", "new": "docker"}]

    def test_extracts_with_surrounding_prose(self) -> None:
        text = 'Вот результат:\n[{"old": "пайтон", "new": "Python"}]\nГотово.'
        items = _extract_items(text)
        assert items == [{"old": "пайтон", "new": "Python"}]

    def test_invalid_returns_empty(self) -> None:
        assert _extract_items("совсем не json") == []

    def test_objects_without_outer_brackets(self) -> None:
        # The model routinely omits the surrounding [ ] and streams objects
        # comma-separated; each complete object must still be recovered.
        text = ' {"old": "комит", "new": "коммит"},\n {"old": "удоли", "new": "Удали"}'
        items = _extract_items(text)
        assert [i["old"] for i in items] == ["комит", "удоли"]

    def test_truncated_reply_keeps_complete_objects(self) -> None:
        # Cut off mid-object on max_tokens: the two whole objects survive, the
        # dangling third is dropped — instead of losing the entire batch.
        text = (
            ' {"old": "комит", "new": "коммит", "confidence": "high", "reason": "x"},\n'
            ' {"old": "удоли", "new": "Удали", "confidence": "high", "reason": "y"},\n'
            ' {"old": "симфоне", "new": "Symfony", "confidence": "medium", "reason": "вари'
        )
        items = _extract_items(text)
        assert [i["old"] for i in items] == ["комит", "удоли"]

    def test_scrape_fallback_not_used_when_json_valid(self) -> None:
        # A proper array with a reason containing a comma must parse as one array,
        # not be re-scraped — the happy path is unchanged.
        text = '[{"old": "комит", "new": "коммит", "reason": "git, часто"}]'
        items = _extract_items(text)
        assert items == [{"old": "комит", "new": "коммит", "reason": "git, часто"}]

    def test_normalize_accepts_find_replace_keys(self) -> None:
        suggestion = _normalize_item({"find": "\\bкомит\\b", "replace": "коммит"}, frequency=3)
        assert suggestion is not None
        assert suggestion.old == "комит"  # word-boundary anchors stripped
        assert suggestion.new == "коммит"

    def test_normalize_rejects_noop(self) -> None:
        assert _normalize_item({"old": "docker", "new": "Docker"}, frequency=1) is None


class TestVerifyStage:
    """Second-pass critic drops candidates it marks 'reject'."""

    def _client(self, verify: bool, ask_replies: list[str]) -> "AnthropicClient":
        # build without __init__ so we don't touch the real SDK
        c = AnthropicClient.__new__(AnthropicClient)
        c._model = "m"  # type: ignore[attr-defined]
        c._effort = "high"  # type: ignore[attr-defined]
        c._verify = verify  # type: ignore[attr-defined]
        c._phrase_index = None  # type: ignore[attr-defined]
        replies = iter(ask_replies)
        c._ask = lambda *_args: next(replies)  # type: ignore[attr-defined,assignment]
        return c

    def test_reject_drops_candidate(self) -> None:
        gen = '[{"old": "комит", "new": "коммит", "confidence": "high", "reason": ""}, '
        gen += '{"old": "курсор", "new": "Cursor", "confidence": "high", "reason": ""}]'
        verify = '[{"old": "комит", "new": "коммит", "verdict": "keep"}, '
        verify += '{"old": "курсор", "new": "Cursor", "verdict": "reject"}]'
        client = self._client(verify=True, ask_replies=[gen, verify])
        batch = [ContextWindow(target="комит и курсор", before=(), after=(), frequency=5)]
        result = client.suggest(batch)
        olds = {s.old for s in result}
        assert olds == {"комит"}  # 'курсор' rejected by the critic

    def test_no_verify_keeps_all(self) -> None:
        gen = '[{"old": "курсор", "new": "Cursor", "confidence": "high", "reason": ""}]'
        client = self._client(verify=False, ask_replies=[gen])
        batch = [ContextWindow(target="открой курсор", before=(), after=(), frequency=5)]
        result = client.suggest(batch)
        assert {s.old for s in result} == {"курсор"}  # no second pass, kept

    def test_known_keys_go_into_prompt(self) -> None:
        client = self._client(verify=False, ask_replies=["[]"])
        captured: dict[str, str] = {}
        client._ask = lambda _system, user: captured.setdefault("user", user) or "[]"  # type: ignore[attr-defined]
        batch = [ContextWindow(target="строка", before=(), after=(), frequency=5)]
        client.suggest(batch, known={"комит", "докер"})
        assert "УЖЕ ПОКРЫТЫЕ ОШИБКИ" in captured["user"]
        assert "комит" in captured["user"] and "докер" in captured["user"]


class _FakeResponse:
    def __init__(self, text: str, output_tokens: int = 0) -> None:
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.usage = type("Usage", (), {"output_tokens": output_tokens})()


def _event(type_: str, **attrs: object) -> object:
    """A duck-typed SSE event object with arbitrary attributes."""
    return type("Event", (), {"type": type_, **attrs})()


def _text_start(index: int) -> object:
    block = type("Block", (), {"type": "text"})()
    return _event("content_block_start", index=index, content_block=block)


def _text_delta(index: int, text: str) -> object:
    delta = type("Delta", (), {"type": "text_delta", "text": text})()
    return _event("content_block_delta", index=index, delta=delta)


def _msg_start(input_tokens: int = 0, cache_read: int = 0) -> object:
    usage = type(
        "Usage",
        (),
        {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": 0,
        },
    )()
    message = type("Message", (), {"usage": usage})()
    return _event("message_start", message=message)


def _msg_delta(output_tokens: int, stop_reason: str = "end_turn") -> object:
    usage = type("Usage", (), {"output_tokens": output_tokens})()
    delta = type("Delta", (), {"stop_reason": stop_reason})()
    return _event("message_delta", usage=usage, delta=delta)


class _FakeEventStream:
    """Context manager mimicking client.messages.create(stream=True) — a raw event iterator."""

    def __init__(self, events: list[object]) -> None:
        self._events = events

    def __enter__(self) -> "_FakeEventStream":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._events)


class _FakeMessages:
    def __init__(self, events: list[object] | None = None) -> None:
        self.blocking_calls = 0
        self.stream_calls = 0
        self._events = events or []

    def create(self, **kwargs: object) -> object:
        if kwargs.get("stream"):
            self.stream_calls += 1
            return _FakeEventStream(self._events)
        self.blocking_calls += 1
        return _FakeResponse("streamed=no")


class TestStreamingTransport:
    """The stream flag picks the transport; both yield the same text from the response."""

    def _client(
        self, stream: bool, events: list[object] | None = None
    ) -> tuple["AnthropicClient", _FakeMessages]:
        c = AnthropicClient.__new__(AnthropicClient)
        c._model = "m"  # type: ignore[attr-defined]
        c._effort = "medium"  # type: ignore[attr-defined]
        c._thinking = False  # type: ignore[attr-defined]
        c._max_tokens = 8000  # type: ignore[attr-defined]
        c._stream = stream  # type: ignore[attr-defined]
        c._on_stream_progress = None  # type: ignore[attr-defined]
        c._stream_started = 0.0  # type: ignore[attr-defined]
        messages = _FakeMessages(events=events)
        c._client = type("Client", (), {"messages": messages})()  # type: ignore[attr-defined]
        return c, messages

    def test_blocking_path_uses_blocking_create(self) -> None:
        client, messages = self._client(stream=False)
        assert client._ask("sys", "user") == "streamed=no"
        assert (messages.blocking_calls, messages.stream_calls) == (1, 0)

    def test_streaming_path_uses_streaming_create(self) -> None:
        events = [_msg_start(), _text_start(0), _text_delta(0, "hi"), _msg_delta(5)]
        client, messages = self._client(stream=True, events=events)
        assert client._ask("sys", "user") == "hi"
        assert (messages.blocking_calls, messages.stream_calls) == (0, 1)

    def test_progress_fires_with_final_token_count(self) -> None:
        events = [_msg_start(), _text_start(0), _text_delta(0, "hello"), _msg_delta(42)]
        client, _ = self._client(stream=True, events=events)
        ticks: list[tuple[int, float]] = []
        client.set_stream_progress(lambda tokens, seconds: ticks.append((tokens, seconds)))
        client._ask("sys", "user")
        assert ticks, "expected at least the final progress tick"
        assert ticks[-1][0] == 42  # last tick carries the exact token count from usage


class TestStreamNonContiguousIndices:
    """Regression: the proxy emits content blocks with skipped indices (0, then 2).

    The SDK's own stream accumulator indexes its snapshot by ``event.index`` and grows
    it one entry per content_block_start, so a delta for block 2 (with only 1 block
    started) raised ``IndexError: list index out of range`` and killed every request.
    Our raw-event consumer must be index-agnostic.
    """

    def _client(self, events: list[object]) -> "AnthropicClient":
        c = AnthropicClient.__new__(AnthropicClient)
        c._model = "m"  # type: ignore[attr-defined]
        c._effort = "medium"  # type: ignore[attr-defined]
        c._thinking = False  # type: ignore[attr-defined]
        c._max_tokens = 8000  # type: ignore[attr-defined]
        c._stream = True  # type: ignore[attr-defined]
        c._on_stream_progress = None  # type: ignore[attr-defined]
        c._stream_started = 0.0  # type: ignore[attr-defined]
        messages = _FakeMessages(events=events)
        c._client = type("Client", (), {"messages": messages})()  # type: ignore[attr-defined]
        return c

    def test_skipped_index_does_not_crash_and_text_concatenated(self) -> None:
        # Block 0 then block 2 — index 1 skipped, exactly the shape that crashed the SDK.
        events = [
            _msg_start(),
            _text_start(0),
            _text_delta(0, "["),
            _event("content_block_stop", index=0),
            _text_start(2),
            _text_delta(2, '{"old":"комит",'),
            _text_delta(2, '"new":"коммит"}'),
            _text_delta(2, "]"),
            _event("content_block_stop", index=2),
            _msg_delta(11),
        ]
        client = self._client(events)
        result = client._ask("sys", "user")
        assert result == '[{"old":"комит","new":"коммит"}]'

    def test_delta_before_any_start_is_tolerated(self) -> None:
        # A delta with no preceding content_block_start at all — still index-agnostic.
        events = [_msg_start(), _text_delta(0, "hi"), _msg_delta(3)]
        client = self._client(events)
        assert client._ask("sys", "user") == "hi"

    def test_usage_and_stop_reason_assembled_from_events(self) -> None:
        events = [
            _msg_start(input_tokens=100, cache_read=80),
            _text_start(0),
            _text_delta(0, "x"),
            _msg_delta(7, stop_reason="end_turn"),
        ]
        client = self._client(events)
        response = client._ask_streaming({})
        assert response.usage.input_tokens == 100
        assert response.usage.cache_read_input_tokens == 80
        assert response.usage.output_tokens == 7
        assert response.stop_reason == "end_turn"


class TestEmptyMaxTokensRetry:
    """An empty reply stopped on max_tokens is a stalled generation -> raise to retry.

    Under load the proxy occasionally returns out=0/stop=max_tokens after a long stall.
    Left as an empty result it would silently zero the batch and advance the checkpoint
    past it; raising routes it through _suggest_with_retry's backoff instead.
    """

    def _client(self, events: list[object]) -> "AnthropicClient":
        c = AnthropicClient.__new__(AnthropicClient)
        c._model = "m"  # type: ignore[attr-defined]
        c._effort = "medium"  # type: ignore[attr-defined]
        c._thinking = False  # type: ignore[attr-defined]
        c._max_tokens = 8000  # type: ignore[attr-defined]
        c._stream = True  # type: ignore[attr-defined]
        c._on_stream_progress = None  # type: ignore[attr-defined]
        c._stream_started = 0.0  # type: ignore[attr-defined]
        messages = _FakeMessages(events=events)
        c._client = type("Client", (), {"messages": messages})()  # type: ignore[attr-defined]
        return c

    def test_empty_max_tokens_raises(self) -> None:
        # out=0, stop=max_tokens, no text deltas — the exact shape seen in analyzer.log.
        events = [_msg_start(), _msg_delta(0, stop_reason="max_tokens")]
        client = self._client(events)
        with pytest.raises(RuntimeError, match="max_tokens"):
            client._ask("sys", "user")

    def test_nonempty_max_tokens_is_kept(self) -> None:
        # A truncated-but-non-empty reply is usable: parse the JSON as far as it goes.
        events = [
            _msg_start(),
            _text_start(0),
            _text_delta(0, '[{"old":"комит","new":"коммит"}]'),
            _msg_delta(16, stop_reason="max_tokens"),
        ]
        client = self._client(events)
        assert client._ask("sys", "user") == '[{"old":"комит","new":"коммит"}]'

    def test_empty_end_turn_is_valid_zero_result(self) -> None:
        # Empty + end_turn = the model genuinely found nothing. Not an error.
        events = [_msg_start(), _msg_delta(1, stop_reason="end_turn")]
        client = self._client(events)
        assert client._ask("sys", "user") == ""

    def test_retry_recovers_the_stalled_batch(self) -> None:
        # End-to-end: a stalled empty reply is retried, the retry lands suggestions.
        stalled = [_msg_start(), _msg_delta(0, stop_reason="max_tokens")]
        healthy = [
            _msg_start(),
            _text_start(0),
            _text_delta(0, '[{"old":"комит","new":"коммит","confidence":"high","reason":""}]'),
            _msg_delta(16, stop_reason="end_turn"),
        ]
        c = AnthropicClient.__new__(AnthropicClient)
        c._model = "m"  # type: ignore[attr-defined]
        c._effort = "medium"  # type: ignore[attr-defined]
        c._thinking = False  # type: ignore[attr-defined]
        c._max_tokens = 8000  # type: ignore[attr-defined]
        c._stream = True  # type: ignore[attr-defined]
        c._verify = False  # type: ignore[attr-defined]
        c._phrase_index = None  # type: ignore[attr-defined]
        c._on_stream_progress = None  # type: ignore[attr-defined]
        c._stream_started = 0.0  # type: ignore[attr-defined]
        replies = iter([_FakeEventStream(stalled), _FakeEventStream(healthy)])
        messages = type("M", (), {"create": lambda _self, **_kw: next(replies)})()
        c._client = type("Client", (), {"messages": messages})()  # type: ignore[attr-defined]

        analyzer = ReplacementAnalyzer.__new__(ReplacementAnalyzer)
        analyzer._client = c  # type: ignore[attr-defined]
        analyzer._config = AnalyzerConfig(  # type: ignore[attr-defined]
            send_known=False, retry_backoff=0, max_error_retries=2
        )
        batch = [ContextWindow(target="сделай комит", before=(), after=(), frequency=3)]
        result = analyzer._suggest_with_retry(batch, seen=set(), index=1, total=1, on_retry=None)
        assert [s.old for s in result] == ["комит"]  # first stalled, retry recovered it


def _patch_log_prompts(monkeypatch, value: bool) -> None:
    """Swap the module-level ``settings`` for a copy with the flag set.

    ``settings`` is a frozen dataclass, so its fields can't be assigned; we rebind the
    whole object in the log_analyzer namespace instead (a plain module attribute).
    """
    import dataclasses

    from sheptun.log_analyzer import settings as current

    monkeypatch.setattr(
        "sheptun.log_analyzer.settings",
        dataclasses.replace(current, analyzer_log_prompts=value),
    )


class TestRequestLogging:
    """Prompt/reply logging is gated behind a flag; the unparsable-reply warning is always on."""

    def _client(self, reply: str) -> "AnthropicClient":
        c = AnthropicClient.__new__(AnthropicClient)
        c._model = "m"  # type: ignore[attr-defined]
        c._effort = "medium"  # type: ignore[attr-defined]
        c._thinking = False  # type: ignore[attr-defined]
        c._max_tokens = 8000  # type: ignore[attr-defined]
        c._stream = True  # type: ignore[attr-defined]
        c._verify = False  # type: ignore[attr-defined]
        c._phrase_index = None  # type: ignore[attr-defined]
        events = [
            _msg_start(),
            _text_start(0),
            _text_delta(0, reply),
            _msg_delta(len(reply), stop_reason="end_turn"),
        ]
        messages = _FakeMessages(events=events)
        c._client = type("Client", (), {"messages": messages})()  # type: ignore[attr-defined]
        c._on_stream_progress = None  # type: ignore[attr-defined]
        c._stream_started = 0.0  # type: ignore[attr-defined]
        return c

    def test_flag_off_does_not_log_prompt_or_reply(self, monkeypatch, caplog) -> None:
        _patch_log_prompts(monkeypatch, False)
        client = self._client('[{"old":"a","new":"b"}]')
        with caplog.at_level("DEBUG", logger="sheptun.analyzer"):
            client._ask("SYSTEM-PROMPT-TEXT", "USER-PROMPT-TEXT")
        assert "USER-PROMPT-TEXT" not in caplog.text
        assert "Сырой ответ" not in caplog.text

    def test_flag_on_logs_user_prompt_and_raw_reply(self, monkeypatch, caplog) -> None:
        _patch_log_prompts(monkeypatch, True)
        client = self._client('[{"old":"a","new":"b"}]')
        with caplog.at_level("DEBUG", logger="sheptun.analyzer"):
            client._ask("SYSTEM-PROMPT-TEXT", "USER-PROMPT-TEXT")
        assert "USER-PROMPT-TEXT" in caplog.text  # user prompt logged
        assert '[{"old":"a","new":"b"}]' in caplog.text  # raw reply logged
        assert "SYSTEM-PROMPT-TEXT" not in caplog.text  # system is static, never logged

    def test_unparsable_nonempty_reply_warns_without_flag(self, monkeypatch, caplog) -> None:
        _patch_log_prompts(monkeypatch, False)
        client = self._client("тут точно нет никакого json")
        batch = [ContextWindow(target="строка", before=(), after=(), frequency=1)]
        with caplog.at_level("WARNING", logger="sheptun.analyzer"):
            result = client.suggest(batch)
        assert result == []
        assert "не распарсился" in caplog.text  # visible even with logging flag off

    def test_empty_reply_does_not_warn(self, monkeypatch, caplog) -> None:
        # An empty end_turn reply is a legit zero result, not a parse failure — no warning.
        _patch_log_prompts(monkeypatch, False)
        client = self._client("")
        batch = [ContextWindow(target="строка", before=(), after=(), frequency=1)]
        with caplog.at_level("WARNING", logger="sheptun.analyzer"):
            client.suggest(batch)
        assert "не распарсился" not in caplog.text


class TestPhraseIndex:
    """Frequency lookup over the actual Recognized lines."""

    def test_counts_lines_containing_phrase(self) -> None:
        index = PhraseIndex(["сделай комит", "ещё комит тут", "открой докер"])
        assert index.frequency("комит") == 2
        assert index.frequency("докер") == 1

    def test_case_insensitive(self) -> None:
        index = PhraseIndex(["Открой Докер"])
        assert index.frequency("докер") == 1
        assert index.frequency("ДОКЕР") == 1

    def test_absent_phrase_is_zero(self) -> None:
        index = PhraseIndex(["сделай комит"])
        assert index.frequency("медлвары") == 0

    def test_empty_phrase_is_zero(self) -> None:
        assert PhraseIndex(["что-то"]).frequency("   ") == 0


class TestPhraseIndexGuardsSuggestions:
    """The client uses the index to fix frequency and drop hallucinations."""

    def _client(self, gen: str, index: PhraseIndex | None) -> "AnthropicClient":
        c = AnthropicClient.__new__(AnthropicClient)
        c._model = "m"  # type: ignore[attr-defined]
        c._effort = "high"  # type: ignore[attr-defined]
        c._verify = False  # type: ignore[attr-defined]
        c._phrase_index = index  # type: ignore[attr-defined]
        c._ask = lambda *_args: gen  # type: ignore[attr-defined,assignment]
        return c

    def test_drops_old_not_in_shown_lines(self) -> None:
        # model invents 'медлвары'; the batch only ever showed 'медлвару' -> dropped
        gen = '[{"old": "медлвары", "new": "middleware", "confidence": "high", "reason": ""}]'
        index = PhraseIndex(["говорю про медлвару"])
        client = self._client(gen, index)
        batch = [ContextWindow(target="говорю про медлвару", before=(), after=(), frequency=99)]
        assert client.suggest(batch) == []

    def test_drops_old_in_log_but_not_in_this_batch(self) -> None:
        # 'комит' exists elsewhere in the log, but this batch never showed it ->
        # it is not the phrase we're replacing here, so it is dropped
        gen = '[{"old": "комит", "new": "коммит", "confidence": "high", "reason": ""}]'
        index = PhraseIndex(["сделай комит"])  # present in the log at large...
        client = self._client(gen, index)
        batch = [ContextWindow(target="открой докер", before=(), after=(), frequency=5)]
        assert client.suggest(batch) == []  # ...but absent from the shown lines

    def test_matches_old_in_context_lines_not_only_target(self) -> None:
        # 'комит' appears in a before/after context line, not the target -> still valid
        gen = '[{"old": "комит", "new": "коммит", "confidence": "high", "reason": ""}]'
        index = PhraseIndex(["сделай комит", "открой докер"])
        client = self._client(gen, index)
        batch = [ContextWindow(target="открой докер", before=("сделай комит",), after=())]
        result = client.suggest(batch)
        assert {s.old for s in result} == {"комит"}

    def test_frequency_is_per_word_not_batch_max(self) -> None:
        gen = '[{"old": "комит", "new": "коммит", "confidence": "high", "reason": ""}]'
        index = PhraseIndex(["сделай комит", "ещё комит"])  # 'комит' in 2 lines
        client = self._client(gen, index)
        # batch frequency is a misleading 454; the real per-word count is 2
        batch = [ContextWindow(target="сделай комит", before=(), after=(), frequency=454)]
        result = client.suggest(batch)
        assert len(result) == 1
        assert result[0].frequency == 2  # honest count, not the batch max

    def test_without_index_keeps_batch_frequency(self) -> None:
        gen = '[{"old": "комит", "new": "коммит", "confidence": "high", "reason": ""}]'
        client = self._client(gen, index=None)
        batch = [ContextWindow(target="сделай комит", before=(), after=(), frequency=7)]
        result = client.suggest(batch)
        assert result[0].frequency == 7  # fallback: no index, batch max stands

    def test_analyzer_wires_index_from_log(self, log_path: Path) -> None:
        # end-to-end: prepare_windows must hand the client a working index
        gen = (
            '[{"old": "комит", "new": "коммит", "confidence": "high", "reason": ""},'
            ' {"old": "выдумка", "new": "нечто", "confidence": "high", "reason": ""}]'
        )
        client = self._client(gen, index=None)
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=10)
        analyzer = ReplacementAnalyzer(config, client=client)
        result = analyzer.analyze(log_path, existing={})
        olds = {s.old for s in result.suggestions}
        assert olds == {"комит"}  # 'выдумка' absent from the log -> dropped
        komit = next(s for s in result.suggestions if s.old == "комит")
        assert komit.frequency == 2  # 'сделай комит' appears twice in SAMPLE_LOG


class TestSuggestionWriter:
    def test_report_does_not_touch_replacements(self, tmp_path: Path) -> None:
        report = tmp_path / "suggested.yaml"
        SuggestionWriter().write_report(
            [ReplacementSuggestion("комит", "коммит", "high", "ошибка ASR", 5)], report
        )
        content = report.read_text(encoding="utf-8")
        assert '"комит": "коммит"' in content
        assert "freq=5" in content

    def test_apply_appends_without_duplicates(self, tmp_path: Path) -> None:
        replacements = tmp_path / "replacements.yaml"
        replacements.write_text('# my notes\n"докер": "docker"\n', encoding="utf-8")
        added = SuggestionWriter().apply(
            [
                ReplacementSuggestion("докер", "docker", "high", "", 1),  # dup
                ReplacementSuggestion("комит", "коммит", "high", "ошибка ASR", 42),  # new
            ],
            replacements,
        )
        assert added == 1
        content = replacements.read_text(encoding="utf-8")
        assert "# my notes" in content  # existing comments preserved (append, not rewrite)
        assert '"докер": "docker"' in content
        # new rule carries the reason comment
        assert '"комит": "коммит"  # freq=42, conf=high — ошибка ASR' in content

    def test_apply_output_stays_valid_yaml_with_control_chars(self, tmp_path: Path) -> None:
        import yaml

        replacements = tmp_path / "replacements.yaml"
        replacements.write_text('"докер": "docker"\n', encoding="utf-8")
        # reason from the model contains a control char (\x08) — must not corrupt the file
        SuggestionWriter().apply(
            [ReplacementSuggestion("комит", "коммит", "high", "символ \x08 в reason", 5)],
            replacements,
        )
        # the whole point: the file must still parse
        loaded = yaml.safe_load(replacements.read_text(encoding="utf-8"))
        assert loaded["комит"] == "коммит"

    def test_normalize_item_strips_control_chars(self) -> None:
        s = _normalize_item(
            {"old": "ко\x08мит", "new": "ком\x08мит", "reason": "bad\x08"}, frequency=1
        )
        assert s is not None
        assert "\x08" not in s.old
        assert "\x08" not in s.new
        assert "\x08" not in s.reason


class _OrderedClient:
    """Deterministic parallel client: batches whose target is in ``slow`` sleep so they
    finish out of order; targets in ``fail`` raise. One unique rule per batch."""

    def __init__(self, slow: set[str] | None = None, fail: set[str] | None = None) -> None:
        self._slow = slow or set()
        self._fail = fail or set()

    def suggest(
        self,
        batch: Sequence[ContextWindow],
        known: set[str] | None = None,  # noqa: ARG002
    ) -> list[ReplacementSuggestion]:
        target = batch[0].target
        if target in self._slow:
            time.sleep(0.4)  # completes after later batches
        if target in self._fail:
            raise RuntimeError(f"boom {target}")
        return [ReplacementSuggestion(target, target.upper(), "high", "", 1)]

    def set_phrase_index(self, index: PhraseIndex) -> None:
        del index


class TestParallelCheckpoint:
    """The parallel path advances the saved position only over the contiguous completed
    prefix of batches, so a crash or interrupt never skips an unfinished batch."""

    def _windows(self, n: int) -> list[ContextWindow]:
        return [ContextWindow(target=f"w{i}", before=(), after=()) for i in range(n)]

    def _config(self, **kw: object) -> AnalyzerConfig:
        base: dict[str, object] = {
            "batch_size": 1,
            "concurrency": 5,
            "delay": 0.0,
            "verify": False,
            "min_confidence": "low",
            "retry_backoff": 0.0,
            "max_error_retries": 0,
        }
        base.update(kw)
        return AnalyzerConfig(**base)  # type: ignore[arg-type]

    def test_out_of_order_completion_keeps_prefix_monotonic(self) -> None:
        # Batch 1 is slowest: the position must not jump ahead while it is still running.
        analyzer = ReplacementAnalyzer(self._config(), client=_OrderedClient(slow={"w0"}))
        analyzer._full_total = 10
        positions: list[int] = []
        result = analyzer.analyze_windows(
            self._windows(10), existing={}, on_progress=lambda p: positions.append(p.position)
        )
        assert result.position == 10
        assert len(result.suggestions) == 10
        assert result.interrupted is False
        assert positions == sorted(positions)  # prefix never rolls back

    def test_failed_batch_freezes_position_before_it(self) -> None:
        # Batch 3 (w2) fails: the committed prefix must stop at 2, not skip the gap.
        analyzer = ReplacementAnalyzer(self._config(), client=_OrderedClient(fail={"w2"}))
        analyzer._full_total = 10
        result = analyzer.analyze_windows(self._windows(10), existing={})
        assert result.position == 2
        assert result.interrupted is True
        assert result.aborted_by_user is False  # a proxy error, not Ctrl+C — no force-exit

    def test_keyboard_interrupt_stops_without_raising(self) -> None:
        # Ctrl+C in the main loop: the method must swallow KeyboardInterrupt, return
        # interrupted=True with the committed prefix, and not block or re-raise.
        analyzer = ReplacementAnalyzer(self._config(concurrency=2), client=_OrderedClient())
        analyzer._full_total = 10

        calls = {"n": 0}
        real_emit = analyzer._emit_prefix_progress

        def emit_then_interrupt(*args: object, **kw: object) -> None:
            real_emit(*args, **kw)  # type: ignore[arg-type]
            calls["n"] += 1
            if calls["n"] >= 1:
                raise KeyboardInterrupt  # simulate Ctrl+C arriving in the main thread

        analyzer._emit_prefix_progress = emit_then_interrupt  # type: ignore[method-assign]
        result = analyzer.analyze_windows(self._windows(10), existing={})
        assert result.interrupted is True
        assert result.aborted_by_user is True  # flags Ctrl+C so the CLI force-exits
        assert result.position <= 10  # returned a valid prefix, did not hang or raise
