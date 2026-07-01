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
    def test_since_keeps_only_newer(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(
            context_lines=1, min_freq=1, batch_size=10, since="2026-07-01 12:00:00"
        )
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)
        targets = {w.target for w in windows}
        assert targets == {"открой докер"}  # only the 2026-07-02 line

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


class TestIterationLimitAndCheckpoint:
    def test_max_iterations_limits_batches_and_orders_by_time(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=1, max_iterations=1)
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)
        result = analyzer.analyze_windows(windows, existing={})
        assert result.processed == 1
        assert result.total == len(windows)
        assert client.calls == 1
        # windows sorted by FIRST-occurrence time; earliest is 'сделай комит'
        # (first seen 10:31:15) → that is the checkpoint after one batch
        assert result.checkpoint == "2026-07-01 10:31:15"

    def test_checkpoint_is_latest_when_all_processed(self, log_path: Path) -> None:
        client = _FakeClient([])
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=10)
        analyzer = ReplacementAnalyzer(config, client=client)
        result = analyzer.analyze(log_path, existing={})
        assert result.checkpoint == "2026-07-02 09:00:00"


class TestProgress:
    def test_callback_fires_per_batch_with_fresh_only(self, log_path: Path) -> None:
        client = _FakeClient([ReplacementSuggestion("комит", "коммит", "high", "", 2)])
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=1)
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


class TestCheckpointNoSkip:
    """Regression: with dedup + chronological order + a line-wise checkpoint, a
    frequent phrase collapses into one window whose timestamp is its LAST
    occurrence. Advancing the checkpoint past it must not skip earlier lines of
    other phrases that were never processed."""

    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        p = tmp_path / "sheptun.log"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_window_timestamp_is_first_occurrence(self, tmp_path: Path) -> None:
        # A frequent phrase occurs at 10:00 and again at 20:00. For a line-wise
        # checkpoint to be safe, the window must be ordered/checkpointed by its
        # FIRST occurrence (10:00), not its last (20:00) — otherwise processing it
        # advances the checkpoint past lines that were never analysed.
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
            "a line-wise checkpoint would skip lines between the two occurrences"
        )


class TestCheckpointOnInterrupt:
    """A long run interrupted mid-way must still advance the checkpoint for the
    batches already processed, so the next run resumes instead of restarting."""

    def test_progress_exposes_running_checkpoint(self, log_path: Path) -> None:
        client = _FakeClient([])
        # chronological order (max_iterations>0) so processed prefix is contiguous
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=1, max_iterations=10)
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)

        checkpoints: list[str] = []
        analyzer.analyze_windows(
            windows, existing={}, on_progress=lambda p: checkpoints.append(p.checkpoint)
        )
        # checkpoint must grow monotonically and match the last window's time
        assert checkpoints == sorted(checkpoints)
        assert checkpoints[-1] == max(w.timestamp for w in windows)

    def test_checkpoint_saved_before_interrupt(self, log_path: Path) -> None:
        client = _FailingClient(fail_on=2)  # 1 batch ok, 2nd raises
        config = AnalyzerConfig(context_lines=1, min_freq=1, batch_size=1, max_iterations=10)
        analyzer = ReplacementAnalyzer(config, client=client)
        windows = analyzer.prepare_windows(log_path)

        saved: list[str] = []

        def on_progress(p: BatchProgress) -> None:
            saved.append(p.checkpoint)  # caller persists after each batch

        with pytest.raises(KeyboardInterrupt):
            analyzer.analyze_windows(windows, existing={}, on_progress=on_progress)

        # the first batch's checkpoint was delivered before the crash → not lost
        assert saved == [windows[0].timestamp]


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
        assert state.last_timestamp() is None
        state.save("2026-07-01 10:00:00")
        assert state.last_timestamp() == "2026-07-01 10:00:00"

    def test_reset(self, tmp_path: Path) -> None:
        state = AnalyzerState(tmp_path / "state.json")
        state.save("2026-07-01 10:00:00")
        state.reset()
        assert state.last_timestamp() is None

    def test_ignores_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("{ not json", encoding="utf-8")
        assert AnalyzerState(path).last_timestamp() is None


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
