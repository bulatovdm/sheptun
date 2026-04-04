import json
from collections.abc import Generator
from pathlib import Path

import pytest

from sheptun.verification import (
    TranscriptRecord,
    VerificationDB,
    VerificationResult,
    load_transcripts,
)


@pytest.fixture
def db(tmp_path: Path) -> Generator[VerificationDB]:
    db = VerificationDB(tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture
def sample_records() -> list[TranscriptRecord]:
    return [
        TranscriptRecord(file="test1.wav", text="привет мир", timestamp="2025-01-01T00:00:00"),
        TranscriptRecord(file="test2.wav", text="клод таб", timestamp="2025-01-01T00:00:01"),
        TranscriptRecord(
            file="test3.wav",
            text="скажи тест",
            timestamp="2025-01-01T00:00:02",
            corrected="скажи тест",
        ),
    ]


@pytest.fixture
def sample_result() -> VerificationResult:
    return VerificationResult(
        verified_text="привет мир",
        is_correct=True,
        confidence="high",
        notes="",
    )


@pytest.fixture
def hallucination_result() -> VerificationResult:
    return VerificationResult(
        verified_text="",
        is_correct=False,
        confidence="high",
        notes="repetitive nonsense",
        is_hallucination=True,
    )


class TestVerificationDB:
    def test_insert_pending(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
        inserted = db.insert_pending(sample_records)

        assert inserted == 3

    def test_insert_pending_skips_duplicates(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
        db.insert_pending(sample_records)
        inserted = db.insert_pending(sample_records)

        assert inserted == 0

    def test_get_pending(self, db: VerificationDB, sample_records: list[TranscriptRecord]) -> None:
        db.insert_pending(sample_records)

        pending = db.get_pending()

        assert len(pending) == 3
        assert pending[0].file == "test1.wav"
        assert pending[0].text == "привет мир"

    def test_get_pending_with_limit(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
        db.insert_pending(sample_records)

        pending = db.get_pending(limit=2)

        assert len(pending) == 2

    def test_get_pending_preserves_corrected(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
        db.insert_pending(sample_records)

        pending = db.get_pending()
        record_with_corrected = next(r for r in pending if r.file == "test3.wav")

        assert record_with_corrected.corrected == "скажи тест"

    def test_save_result(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        sample_result: VerificationResult,
    ) -> None:
        db.insert_pending(sample_records)

        db.save_result("test1.wav", sample_result, "claude-haiku-4.5")

        pending = db.get_pending()
        assert len(pending) == 2  # one less pending

    def test_save_result_updates_fields(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
    ) -> None:
        db.insert_pending(sample_records)
        result = VerificationResult(
            verified_text="привет, мир",
            is_correct=False,
            confidence="medium",
            notes="added comma",
        )

        db.save_result("test1.wav", result, "claude-haiku-4.5")

        stats = db.get_stats()
        assert stats["completed"] == 1
        assert stats["fixed"] == 1

    def test_save_error(self, db: VerificationDB, sample_records: list[TranscriptRecord]) -> None:
        db.insert_pending(sample_records)

        db.save_error("test1.wav", "API error")

        stats = db.get_stats()
        assert stats["error"] == 1

    def test_get_stats(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        sample_result: VerificationResult,
    ) -> None:
        db.insert_pending(sample_records)
        db.save_result("test1.wav", sample_result, "claude-haiku-4.5")
        db.save_error("test2.wav", "error")

        stats = db.get_stats()

        assert stats["total"] == 3
        assert stats["completed"] == 1
        assert stats["pending"] == 1
        assert stats["error"] == 1
        assert stats["correct"] == 1
        assert stats["fixed"] == 0

    def test_reset_errors(self, db: VerificationDB, sample_records: list[TranscriptRecord]) -> None:
        db.insert_pending(sample_records)
        db.save_error("test1.wav", "API error")
        db.save_error("test2.wav", "timeout")

        count = db.reset_errors()

        assert count == 2
        pending = db.get_pending()
        assert len(pending) == 3  # all back to pending

    def test_reset_errors_returns_zero_when_no_errors(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
        db.insert_pending(sample_records)

        count = db.reset_errors()

        assert count == 0

    def test_reset_all(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        sample_result: VerificationResult,
    ) -> None:
        db.insert_pending(sample_records)
        db.save_result("test1.wav", sample_result, "claude-haiku-4.5")
        db.save_error("test2.wav", "error")

        count = db.reset_all()

        assert count == 2
        pending = db.get_pending()
        assert len(pending) == 3

    def test_reset_all_clears_fields(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        sample_result: VerificationResult,
    ) -> None:
        db.insert_pending(sample_records)
        db.save_result("test1.wav", sample_result, "claude-haiku-4.5")

        db.reset_all()

        stats = db.get_stats()
        assert stats.get("completed", 0) == 0
        assert stats.get("error", 0) == 0
        assert stats["pending"] == 3

    def test_export_jsonl(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        sample_result: VerificationResult,
        tmp_path: Path,
    ) -> None:
        db.insert_pending(sample_records)
        db.save_result("test1.wav", sample_result, "claude-haiku-4.5")

        output = tmp_path / "export.jsonl"
        count = db.export_jsonl(output)

        assert count == 1
        with output.open() as f:
            data = json.loads(f.readline())
        assert data["file"] == "test1.wav"
        assert data["verified_text"] == "привет мир"
        assert data["is_correct"] is True

    def test_save_hallucination(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        hallucination_result: VerificationResult,
    ) -> None:
        db.insert_pending(sample_records)

        db.save_result("test1.wav", hallucination_result, "claude-haiku-4.5")

        stats = db.get_stats()
        assert stats["hallucinations"] == 1
        assert stats["correct"] == 0
        assert stats["fixed"] == 0

    def test_hallucinations_not_in_correct_or_fixed(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        hallucination_result: VerificationResult,
    ) -> None:
        db.insert_pending(sample_records)
        db.save_result("test1.wav", hallucination_result, "claude-haiku-4.5")

        stats = db.get_stats()

        assert stats["hallucinations"] == 1
        assert stats["correct"] == 0
        assert stats["fixed"] == 0
        assert stats["completed"] == 1

    def test_export_excludes_hallucinations_by_default(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        sample_result: VerificationResult,
        hallucination_result: VerificationResult,
        tmp_path: Path,
    ) -> None:
        db.insert_pending(sample_records)
        db.save_result("test1.wav", sample_result, "claude-haiku-4.5")
        db.save_result("test2.wav", hallucination_result, "claude-haiku-4.5")

        output = tmp_path / "export.jsonl"
        count = db.export_jsonl(output)

        assert count == 1

    def test_export_includes_hallucinations_when_requested(
        self,
        db: VerificationDB,
        sample_records: list[TranscriptRecord],
        sample_result: VerificationResult,
        hallucination_result: VerificationResult,
        tmp_path: Path,
    ) -> None:
        db.insert_pending(sample_records)
        db.save_result("test1.wav", sample_result, "claude-haiku-4.5")
        db.save_result("test2.wav", hallucination_result, "claude-haiku-4.5")

        output = tmp_path / "export.jsonl"
        count = db.export_jsonl(output, exclude_hallucinations=False)

        assert count == 2

    def test_schema_migration_adds_hallucination_column(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE verifications ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "file TEXT NOT NULL UNIQUE, "
            "original_text TEXT NOT NULL, "
            "corrected_text TEXT, "
            "verified_text TEXT, "
            "is_correct INTEGER, "
            "confidence TEXT, "
            "notes TEXT, "
            "model TEXT, "
            "status TEXT NOT NULL DEFAULT 'pending', "
            "error_message TEXT, "
            "created_at TEXT NOT NULL, "
            "completed_at TEXT)"
        )
        conn.commit()
        conn.close()

        db = VerificationDB(db_path)
        db.close()

        check = sqlite3.connect(str(db_path))
        columns = {row[1] for row in check.execute("PRAGMA table_info(verifications)").fetchall()}
        check.close()
        assert "is_hallucination" in columns

    def test_migration_resets_broken_records(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE verifications ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "file TEXT NOT NULL UNIQUE, "
            "original_text TEXT NOT NULL, "
            "corrected_text TEXT, "
            "verified_text TEXT, "
            "is_correct INTEGER, "
            "confidence TEXT, "
            "notes TEXT, "
            "model TEXT, "
            "status TEXT NOT NULL DEFAULT 'pending', "
            "error_message TEXT, "
            "created_at TEXT NOT NULL, "
            "completed_at TEXT)"
        )
        conn.execute(
            "INSERT INTO verifications (file, original_text, verified_text, is_correct, "
            "notes, status, created_at) VALUES "
            "('broken.wav', 'мусор', '[REJECT - INVALID]', 0, "
            "'should be rejected', 'completed', '2025-01-01')"
        )
        conn.execute(
            "INSERT INTO verifications (file, original_text, verified_text, is_correct, "
            "notes, status, created_at) VALUES "
            "('good.wav', 'привет', 'привет', 1, '', 'completed', '2025-01-01')"
        )
        conn.commit()
        conn.close()

        db = VerificationDB(db_path)
        try:
            pending = db.get_pending()
            assert len(pending) == 1
            assert pending[0].file == "broken.wav"
        finally:
            db.close()


class TestLoadTranscripts:
    def test_load_basic(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "transcripts.jsonl"
        jsonl.write_text(
            json.dumps({"file": "a.wav", "text": "тест", "timestamp": "2025-01-01"})
            + "\n"
            + json.dumps({"file": "b.wav", "text": "ещё", "timestamp": "2025-01-02"})
            + "\n"
        )

        records = load_transcripts(jsonl)

        assert len(records) == 2
        assert records[0].file == "a.wav"
        assert records[0].text == "тест"

    def test_load_with_corrected_field(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "transcripts.jsonl"
        jsonl.write_text(
            json.dumps(
                {"file": "a.wav", "text": "тест", "timestamp": "2025-01-01", "corrected": "тест!"}
            )
            + "\n"
        )

        records = load_transcripts(jsonl)

        assert records[0].corrected == "тест!"

    def test_load_skips_empty_lines(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "transcripts.jsonl"
        jsonl.write_text(
            json.dumps({"file": "a.wav", "text": "тест", "timestamp": "2025-01-01"}) + "\n\n\n"
        )

        records = load_transcripts(jsonl)

        assert len(records) == 1


