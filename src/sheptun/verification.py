from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console

from sheptun.settings import settings

if TYPE_CHECKING:
    from pathlib import Path

SYSTEM_PROMPT = """\
You are an expert in Russian speech recognition quality control.
You verify and correct transcriptions from Whisper speech recognition
used in a voice-controlled terminal application called Sheptun.

Context about the application:
- Users speak Russian to control a macOS terminal
- Common voice commands (said in Russian, mapped to actions):
  "таб", "шифт таб", "энтер", "ввод", "эскейп", "пробел",
  "вверх", "вниз", "влево", "вправо", "удалить", "удали",
  "удали слово", "удали строку", "бэкспейс",
  "копировать", "вставить", "вырезать", "отменить", "сохранить",
  "выделить всё", "контрол си", "контрол зэт",
  "хоум", "энд", "клод" (means "claude")
- Slash commands: "слэш модель", "слэш хелп", "слэш клир", "слэш компакт",
  "слэш память", "слэш контекст", "слэш конфиг", "слэш ревью", "слэш баг"
- Dictation prefixes: "скажи", "введи", "напиши", "текст"
- Stop commands: "шептун стоп", "шептун хватит", "шептун выход"
- Users discuss programming, git, APIs, Docker, testing, deployment
- Mixed Russian/English vocabulary is normal (e.g., "commit", "deploy", "API")
- Short phrases (1-5 words) are common voice commands

CRITICAL rules for English terms and abbreviations:
- Keep English terms, file names, and abbreviations in their original form
- "CLAUDE.md" must stay as "CLAUDE.md", NOT "клауд мд"
- "README" must stay as "README", NOT "ридми"
- "Docker", "API", "git", "commit", "deploy", "pytest", "pip" — keep in English
- File paths and extensions: ".py", ".yaml", ".md", ".env" — keep as-is
- Programming terms: "fine-tuning", "dataset", "frontend", "backend" — keep in English
- If Whisper wrote an English term in Russian transliteration, convert it BACK to English

Common Whisper errors to look for:
1. Word boundary errors (words incorrectly split or merged)
2. Hallucinated text (repetitive nonsense, foreign scripts, YouTube-style endings)
3. Wrong words due to phonetic similarity
4. Missing or incorrect punctuation
5. Incorrect capitalization
6. English terms incorrectly transliterated to Russian (e.g., "клауд" → "Claude")

IMPORTANT: Respond ONLY with valid JSON, no other text:
{
  "verified_text": "corrected text or original if correct",
  "is_correct": true,
  "confidence": "high",
  "notes": "brief explanation of changes, or empty string if correct"
}"""

DB_SCHEMA = """\
CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file TEXT NOT NULL UNIQUE,
    original_text TEXT NOT NULL,
    corrected_text TEXT,
    verified_text TEXT,
    is_correct INTEGER,
    confidence TEXT,
    notes TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
)"""

console = Console()


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
        self._conn.commit()

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
            "confidence = ?, notes = ?, model = ?, status = 'completed', "
            "completed_at = ? WHERE file = ?",
            (
                result.verified_text,
                1 if result.is_correct else 0,
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
            "is_correct = NULL, confidence = NULL, notes = NULL, model = NULL, "
            "error_message = NULL, completed_at = NULL "
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
            "SELECT COUNT(*) as cnt FROM verifications WHERE is_correct = 1"
        ).fetchone()
        fixed = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM verifications WHERE is_correct = 0 AND status = 'completed'"
        ).fetchone()
        stats["correct"] = correct["cnt"] if correct else 0
        stats["fixed"] = fixed["cnt"] if fixed else 0
        stats["total"] = sum(v for k, v in stats.items() if k in ("pending", "completed", "error"))
        return stats

    def export_jsonl(self, output_path: Path) -> int:
        rows = self._conn.execute(
            "SELECT file, original_text, verified_text, is_correct, confidence, notes, model "
            "FROM verifications WHERE status = 'completed' ORDER BY file"
        ).fetchall()
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                entry: dict[str, str | bool] = {
                    "file": row["file"],
                    "original_text": row["original_text"],
                    "verified_text": row["verified_text"],
                    "is_correct": bool(row["is_correct"]),
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


class ClaudeVerifier:
    def __init__(self, model: str | None = None) -> None:
        self._model = model

    async def verify_single(self, record: TranscriptRecord) -> tuple[VerificationResult, str]:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

        prompt = self.build_prompt(record)
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=1,
            model=self._model,
        )

        response_text = ""
        model_used = self._model or "default"
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                model_used = message.model
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        return self.parse_response(response_text), model_used

    def build_prompt(self, record: TranscriptRecord) -> str:
        if record.corrected:
            return (
                f"Verify this Russian speech transcription.\n"
                f'Original: "{record.text}"\n'
                f'Spell-corrected: "{record.corrected}"'
            )
        return f'Verify this Russian speech transcription: "{record.text}"'

    def parse_response(self, text: str) -> VerificationResult:
        cleaned = text.strip()

        # Strip markdown code block
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Extract first JSON object {...} from response
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        data = json.loads(cleaned)
        return VerificationResult(
            verified_text=data["verified_text"],
            is_correct=bool(data["is_correct"]),
            confidence=data.get("confidence", "medium"),
            notes=data.get("notes", ""),
        )


async def run_verification(
    dataset_path: Path | None = None,
    limit: int | None = None,
    model: str | None = None,
    concurrency: int = 1,
) -> None:
    ds_path = dataset_path or settings.dataset_path
    transcripts_file = ds_path / "transcripts.jsonl"

    if not transcripts_file.exists():
        console.print("[red]Файл транскрипций не найден[/red]")
        return

    db = VerificationDB(ds_path / "verification.db")

    try:
        console.print("[yellow]Загрузка транскрипций...[/yellow]")
        records = load_transcripts(transcripts_file)
        inserted = db.insert_pending(records)
        if inserted > 0:
            console.print(f"[green]Добавлено {inserted} новых записей в БД[/green]")

        pending = db.get_pending(limit)
        if not pending:
            console.print("[green]Нет записей для обработки[/green]")
            return

        total = len(pending)
        if concurrency > 1:
            console.print(
                f"[yellow]Обработка {total} записей ({concurrency} потоков)...[/yellow]\n"
            )
        else:
            console.print(f"[yellow]Обработка {total} записей...[/yellow]\n")

        verifier = ClaudeVerifier(model=model)
        completed_count = 0

        async def process_record(record: TranscriptRecord) -> None:
            nonlocal completed_count
            try:
                result, model_used = await verifier.verify_single(record)
                db.save_result(record.file, result, model_used)
                completed_count += 1

                status = "[green]OK[/green]" if result.is_correct else "[yellow]FIXED[/yellow]"
                console.print(f"[{completed_count}/{total}] {status} {record.file}")
                if not result.is_correct:
                    console.print(f"  [dim]{record.text}[/dim]")
                    console.print(f"  [cyan]{result.verified_text}[/cyan]")
                    if result.notes:
                        console.print(f"  [dim italic]{result.notes}[/dim italic]")
            except Exception as e:
                db.save_error(record.file, str(e))
                completed_count += 1
                console.print(f"[{completed_count}/{total}] [red]ERROR[/red] {record.file}: {e}")

        if concurrency <= 1:
            for record in pending:
                await process_record(record)
        else:
            semaphore = asyncio.Semaphore(concurrency)

            async def limited(record: TranscriptRecord) -> None:
                async with semaphore:
                    await process_record(record)

            await asyncio.gather(*(limited(r) for r in pending))

        console.print()
        stats = db.get_stats()
        console.print(
            f"[bold]Готово:[/bold] {stats.get('completed', 0)} обработано, "
            f"{stats.get('correct', 0)} верных, {stats.get('fixed', 0)} исправлено, "
            f"{stats.get('error', 0)} ошибок"
        )
    finally:
        db.close()
