from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sheptun.settings import settings

if TYPE_CHECKING:
    from pathlib import Path

DB_SCHEMA = """\
CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file TEXT NOT NULL UNIQUE,
    original_text TEXT NOT NULL,
    corrected_text TEXT,
    verified_text TEXT,
    is_correct INTEGER,
    is_hallucination INTEGER DEFAULT 0,
    confidence TEXT,
    notes TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
)"""


@dataclass(frozen=True)
class TranscriptRecord:
    file: str
    text: str
    timestamp: str
    corrected: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    verified_text: str
    is_correct: bool
    confidence: str
    notes: str
    is_hallucination: bool = False


class VerificationDB:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or (settings.dataset_path / "verification.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(DB_SCHEMA)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_verifications_status ON verifications(status)"
        )
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(verifications)").fetchall()
        }
        if "is_hallucination" not in columns:
            self._conn.execute(
                "ALTER TABLE verifications ADD COLUMN is_hallucination INTEGER DEFAULT 0"
            )
            self._reset_broken_records()

    def _reset_broken_records(self) -> None:
        self._conn.execute(
            "UPDATE verifications SET status = 'pending', verified_text = NULL, "
            "is_correct = NULL, confidence = NULL, notes = NULL, model = NULL, "
            "error_message = NULL, completed_at = NULL "
            "WHERE status = 'completed' AND ("
            "  verified_text LIKE '%[REJECT%' OR "
            "  length(verified_text) > 500 OR "
            "  notes LIKE '%severely corrupted%' OR "
            "  notes LIKE '%should be rejected%' OR "
            "  notes LIKE '%unusable%' OR "
            "  notes LIKE '%not salvageable%' OR "
            "  notes LIKE '%cannot be reliably%' OR "
            "  notes LIKE '%impossible to reliably%' OR "
            "  notes LIKE '%impossible to correct%' OR "
            "  notes LIKE '%should be re-recorded%' OR "
            "  notes LIKE '%Recommend re-recording%' OR "
            "  notes LIKE '%severe recognition failure%' OR "
            "  notes LIKE '%extensive Whisper hallucinations%'"
            ")"
        )

    def close(self) -> None:
        self._conn.close()

    def insert_pending(self, records: list[TranscriptRecord]) -> int:
        inserted = 0
        now = datetime.now().isoformat()
        for record in records:
            try:
                self._conn.execute(
                    "INSERT INTO verifications (file, original_text, corrected_text, status, created_at) "
                    "VALUES (?, ?, ?, 'pending', ?)",
                    (record.file, record.text, record.corrected, now),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass
        self._conn.commit()
        return inserted

    def get_pending(self, limit: int | None = None) -> list[TranscriptRecord]:
        query = (
            "SELECT file, original_text, corrected_text FROM verifications WHERE status = 'pending'"
        )
        if limit:
            query += f" LIMIT {limit}"
        rows = self._conn.execute(query).fetchall()
        return [
            TranscriptRecord(
                file=row["file"],
                text=row["original_text"],
                timestamp="",
                corrected=row["corrected_text"],
            )
            for row in rows
        ]

    def save_result(self, file: str, result: VerificationResult, model: str) -> None:
        self._conn.execute(
            "UPDATE verifications SET verified_text = ?, is_correct = ?, "
            "is_hallucination = ?, confidence = ?, notes = ?, model = ?, "
            "status = 'completed', completed_at = ? WHERE file = ?",
            (
                result.verified_text,
                1 if result.is_correct else 0,
                1 if result.is_hallucination else 0,
                result.confidence,
                result.notes,
                model,
                datetime.now().isoformat(),
                file,
            ),
        )
        self._conn.commit()

    def save_error(self, file: str, error: str) -> None:
        self._conn.execute(
            "UPDATE verifications SET status = 'error', error_message = ?, "
            "completed_at = ? WHERE file = ?",
            (error, datetime.now().isoformat(), file),
        )
        self._conn.commit()

    def reset_errors(self) -> int:
        cursor = self._conn.execute(
            "UPDATE verifications SET status = 'pending', error_message = NULL, "
            "completed_at = NULL WHERE status = 'error'"
        )
        self._conn.commit()
        return cursor.rowcount

    def reset_all(self) -> int:
        cursor = self._conn.execute(
            "UPDATE verifications SET status = 'pending', verified_text = NULL, "
            "is_correct = NULL, is_hallucination = NULL, confidence = NULL, "
            "notes = NULL, model = NULL, error_message = NULL, completed_at = NULL "
            "WHERE status IN ('completed', 'error')"
        )
        self._conn.commit()
        return cursor.rowcount

    def get_stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM verifications GROUP BY status"
        ).fetchall()
        stats: dict[str, int] = {row["status"]: row["cnt"] for row in rows}
        correct = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM verifications "
            "WHERE is_correct = 1 AND is_hallucination = 0"
        ).fetchone()
        fixed = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM verifications "
            "WHERE is_correct = 0 AND status = 'completed' AND is_hallucination = 0"
        ).fetchone()
        hallucinations = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM verifications WHERE is_hallucination = 1"
        ).fetchone()
        stats["correct"] = correct["cnt"] if correct else 0
        stats["fixed"] = fixed["cnt"] if fixed else 0
        stats["hallucinations"] = hallucinations["cnt"] if hallucinations else 0
        stats["total"] = sum(v for k, v in stats.items() if k in ("pending", "completed", "error"))
        return stats

    def export_jsonl(self, output_path: Path, *, exclude_hallucinations: bool = True) -> int:
        query = (
            "SELECT file, original_text, verified_text, is_correct, "
            "is_hallucination, confidence, notes, model "
            "FROM verifications WHERE status = 'completed'"
        )
        if exclude_hallucinations:
            query += " AND is_hallucination = 0"
        query += " ORDER BY file"
        rows = self._conn.execute(query).fetchall()
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                entry: dict[str, str | bool] = {
                    "file": row["file"],
                    "original_text": row["original_text"],
                    "verified_text": row["verified_text"],
                    "is_correct": bool(row["is_correct"]),
                    "is_hallucination": bool(row["is_hallucination"]),
                    "confidence": row["confidence"],
                    "notes": row["notes"],
                    "model": row["model"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return len(rows)


def load_transcripts(transcripts_path: Path) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    with transcripts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                TranscriptRecord(
                    file=data["file"],
                    text=data["text"],
                    timestamp=data.get("timestamp", ""),
                    corrected=data.get("corrected"),
                )
            )
    return records


