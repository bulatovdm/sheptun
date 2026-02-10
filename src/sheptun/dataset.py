import json
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from sheptun.settings import settings

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1


class DatasetRecorder:
    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or settings.dataset_path
        self.audio_dir = self.dataset_path / "audio"
        self.transcripts_file = self.dataset_path / "transcripts.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def save(self, audio: np.ndarray, text: str, corrected_text: str | None = None) -> Path:
        filename = self._generate_filename()
        audio_path = self.audio_dir / f"{filename}.wav"

        self._save_wav(audio_path, audio)
        self._append_transcript(filename, text, corrected_text)

        return audio_path

    def _save_wav(self, path: Path, audio: np.ndarray) -> None:
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

    def _append_transcript(
        self, filename: str, text: str, corrected_text: str | None = None
    ) -> None:
        record: dict[str, str] = {
            "file": f"{filename}.wav",
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }
        if corrected_text is not None:
            record["corrected"] = corrected_text

        with self.transcripts_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_stats(self) -> dict[str, int]:
        audio_count = len(list(self.audio_dir.glob("*.wav")))
        transcript_count = 0

        if self.transcripts_file.exists():
            with self.transcripts_file.open(encoding="utf-8") as f:
                transcript_count = sum(1 for _ in f)

        return {"audio_files": audio_count, "transcripts": transcript_count}

    def clear(self) -> None:
        for audio_file in self.audio_dir.glob("*.wav"):
            audio_file.unlink()

        if self.transcripts_file.exists():
            self.transcripts_file.unlink()
