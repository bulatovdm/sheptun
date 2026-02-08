import json
from collections.abc import Generator
from pathlib import Path

import pytest

from sheptun.verification import (
    ClaudeVerifier,
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

    def test_get_pending(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
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

    def test_save_error(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
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

    def test_reset_errors(
        self, db: VerificationDB, sample_records: list[TranscriptRecord]
    ) -> None:
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
            json.dumps({"file": "a.wav", "text": "тест", "timestamp": "2025-01-01"})
            + "\n\n\n"
        )

        records = load_transcripts(jsonl)

        assert len(records) == 1


class TestClaudeVerifier:
    def test_build_prompt_simple(self) -> None:
        verifier = ClaudeVerifier()
        record = TranscriptRecord(file="test.wav", text="привет", timestamp="")

        prompt = verifier.build_prompt(record)

        assert prompt == 'Verify this Russian speech transcription: "привет"'

    def test_build_prompt_with_corrected(self) -> None:
        verifier = ClaudeVerifier()
        record = TranscriptRecord(
            file="test.wav", text="привет", timestamp="", corrected="Привет"
        )

        prompt = verifier.build_prompt(record)

        assert 'Original: "привет"' in prompt
        assert 'Spell-corrected: "Привет"' in prompt

    def test_parse_response_valid_json(self) -> None:
        verifier = ClaudeVerifier()
        response = json.dumps(
            {
                "verified_text": "привет мир",
                "is_correct": True,
                "confidence": "high",
                "notes": "",
            }
        )

        result = verifier.parse_response(response)

        assert result.verified_text == "привет мир"
        assert result.is_correct is True
        assert result.confidence == "high"

    def test_parse_response_with_code_block(self) -> None:
        verifier = ClaudeVerifier()
        response = '```json\n{"verified_text": "тест", "is_correct": true, "confidence": "high", "notes": ""}\n```'

        result = verifier.parse_response(response)

        assert result.verified_text == "тест"

    def test_parse_response_is_correct_false(self) -> None:
        verifier = ClaudeVerifier()
        response = json.dumps(
            {
                "verified_text": "привет, мир",
                "is_correct": False,
                "confidence": "medium",
                "notes": "added comma",
            }
        )

        result = verifier.parse_response(response)

        assert result.is_correct is False
        assert result.notes == "added comma"

    def test_parse_response_json_with_surrounding_text(self) -> None:
        verifier = ClaudeVerifier()
        response = (
            'Here is my analysis:\n'
            '{"verified_text": "тест", "is_correct": true, "confidence": "high", "notes": ""}\n'
            'Hope this helps!'
        )

        result = verifier.parse_response(response)

        assert result.verified_text == "тест"
        assert result.is_correct is True

    def test_parse_response_json_with_extra_data(self) -> None:
        verifier = ClaudeVerifier()
        response = (
            '{"verified_text": "тест", "is_correct": true, "confidence": "high", "notes": ""}\n'
            '{"verified_text": "extra", "is_correct": false, "confidence": "low", "notes": "dup"}'
        )

        result = verifier.parse_response(response)

        assert result.verified_text == "тест"

    def test_parse_response_invalid_json_raises(self) -> None:
        verifier = ClaudeVerifier()

        with pytest.raises(json.JSONDecodeError):
            verifier.parse_response("not json at all")
