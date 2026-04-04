import json
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sheptun.types import SpeechRecognizer

console = Console()

_MODEL_LABELS: dict[str, str] = {
    "mlx": "MLX Whisper",
    "whisper": "Whisper (CPU)",
    "apple": "Apple Speech",
    "parakeet": "Parakeet TDT",
    "qwen": "Qwen3-ASR",
}

DEFAULT_MODELS = ["mlx", "whisper", "apple", "parakeet"]
DEFAULT_N_FILES = 5

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass
class FileResult:
    filename: str
    text: str | None
    reference: str | None
    inference_time: float
    audio_duration: float
    rtf: float
    cer_norm: float | None = None   # CER после нормализации (без пунктуации, lowercase)
    cer_raw: float | None = None    # CER with punctuation


@dataclass
class BenchmarkResult:
    model_name: str
    load_time: float
    file_results: list[FileResult] = field(default_factory=list)

    @property
    def avg_rtf(self) -> float:
        if not self.file_results:
            return 0.0
        return sum(r.rtf for r in self.file_results) / len(self.file_results)

    @property
    def avg_inference_time(self) -> float:
        if not self.file_results:
            return 0.0
        return sum(r.inference_time for r in self.file_results) / len(self.file_results)

    @property
    def avg_cer_norm(self) -> float | None:
        values = [r.cer_norm for r in self.file_results if r.cer_norm is not None]
        return sum(values) / len(values) if values else None

    @property
    def avg_cer_raw(self) -> float | None:
        values = [r.cer_raw for r in self.file_results if r.cer_raw is not None]
        return sum(values) / len(values) if values else None


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    return " ".join(text.split())


def _compute_cer(hypothesis: str, reference: str) -> float:
    try:
        import jiwer  # type: ignore[import-untyped]

        return jiwer.cer(reference, hypothesis)  # type: ignore[return-value]
    except ImportError:
        return _cer_fallback(hypothesis, reference)


def _cer_fallback(hypothesis: str, reference: str) -> float:
    """Simple edit distance CER without jiwer."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    n = len(ref_chars)
    if n == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    # Levenshtein distance
    dp = list(range(len(hyp_chars) + 1))
    for i, rc in enumerate(ref_chars):
        new_dp = [i + 1]
        for j, hc in enumerate(hyp_chars):
            cost = 0 if rc == hc else 1
            new_dp.append(min(new_dp[j] + 1, dp[j + 1] + 1, dp[j] + cost))
        dp = new_dp
    return dp[len(hyp_chars)] / n


def _get_wav_duration(path: Path) -> float:
    with wave.open(str(path)) as wf:
        return int(wf.getnframes()) / float(wf.getframerate())


def _load_wav_as_bytes(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path)) as wf:
        sample_rate = wf.getframerate()
        raw_bytes = wf.readframes(wf.getnframes())
    return raw_bytes, sample_rate


def load_references(transcripts_file: Path) -> dict[str, str]:
    """Загрузить эталонные транскрипции из .jsonl файла.

    Поддерживает два формата:
    - {"file": "name.wav", "text": "..."}  — датасет транскрипций
    - {"id": "01_name", "text": "..."}     — тест-сет (id → name.wav)
    """
    if not transcripts_file.exists():
        return {}
    refs: dict[str, str] = {}
    with transcripts_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("corrected") or record.get("text", "")
                if not text:
                    continue
                # Format 1: {"file": "name.wav", ...}
                if "file" in record:
                    refs[record["file"]] = text
                # Format 2: {"id": "01_name", ...} → key is "01_name.wav"
                elif "id" in record:
                    refs[f"{record['id']}.wav"] = text
            except json.JSONDecodeError:
                pass
    return refs


def _parse_model_key(model_key: str) -> tuple[str, str | None]:
    """Parse 'whisper:turbo' → ('whisper', 'turbo'), 'mlx' → ('mlx', None)."""
    if ":" in model_key:
        engine, model = model_key.split(":", 1)
        return engine.strip(), model.strip()
    return model_key, None


def _load_recognizer(model_key: str) -> SpeechRecognizer | None:
    engine, model = _parse_model_key(model_key)
    try:
        if engine == "mlx":
            from sheptun.recognition import MLXWhisperRecognizer

            return MLXWhisperRecognizer(model_name=model or "turbo")

        if engine == "whisper":
            from sheptun.recognition import WhisperRecognizer

            return WhisperRecognizer(model_name=model or "base")

        if engine == "apple":
            from sheptun.apple_speech import AppleSpeechRecognizer

            return AppleSpeechRecognizer()

        if engine == "parakeet":
            from sheptun.parakeet import ParakeetRecognizer

            return ParakeetRecognizer()

        if engine == "qwen":
            from sheptun.qwen_asr import QwenASRRecognizer

            return QwenASRRecognizer(model_id=model or "Qwen/Qwen3-ASR-0.6B")

    except ImportError as e:
        console.print(f"[yellow]  Пропуск {model_key}: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]  Ошибка загрузки {model_key}: {e}[/red]")
    return None


def _benchmark_model(
    model_key: str,
    audio_files: list[Path],
    references: dict[str, str],
) -> BenchmarkResult | None:
    engine, model = _parse_model_key(model_key)
    base_label = _MODEL_LABELS.get(engine, engine)
    label = f"{base_label} ({model})" if model else base_label
    console.print(f"\n[bold cyan]Модель: {label}[/bold cyan]")
    console.print("  Загрузка...", end="")

    t0 = time.perf_counter()
    recognizer = _load_recognizer(model_key)
    load_time = time.perf_counter() - t0

    if recognizer is None:
        return None

    console.print(f" загружена за [green]{load_time:.2f}s[/green]")
    result = BenchmarkResult(model_name=label, load_time=load_time)

    for wav_path in audio_files:
        audio_bytes, sample_rate = _load_wav_as_bytes(wav_path)
        audio_duration = _get_wav_duration(wav_path)
        reference = references.get(wav_path.name)

        t_start = time.perf_counter()
        recognition = recognizer.recognize(audio_bytes, sample_rate)
        inference_time = time.perf_counter() - t_start

        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
        text = recognition.text if recognition else None

        cer_norm: float | None = None
        cer_raw: float | None = None
        if text is not None and reference is not None:
            cer_norm = _compute_cer(_normalize(text), _normalize(reference))
            cer_raw = _compute_cer(text, reference)

        file_result = FileResult(
            filename=wav_path.name,
            text=text,
            reference=reference,
            inference_time=inference_time,
            audio_duration=audio_duration,
            rtf=rtf,
            cer_norm=cer_norm,
            cer_raw=cer_raw,
        )
        result.file_results.append(file_result)

        cer_str = _cer_display(cer_norm, cer_raw) if reference else ""
        status = f"[dim]{text[:50]}[/dim]" if text else "[red](нет результата)[/red]"
        console.print(
            f"  {wav_path.name:<30} {inference_time:.3f}s  RTF={rtf:.2f}{cer_str}  {status}"
        )

    return result


def _cer_display(cer_norm: float | None, cer_raw: float | None) -> str:
    if cer_norm is None:
        return ""
    style = "green" if cer_norm < 0.1 else "yellow" if cer_norm < 0.3 else "red"
    raw_str = f"/{cer_raw:.0%}" if cer_raw is not None else ""
    return f"  CER=[{style}]{cer_norm:.0%}{raw_str}[/{style}]"


def _rtf_style(rtf: float) -> str:
    if rtf < 1.0:
        return "green"
    if rtf < 2.0:
        return "yellow"
    return "red"


def _cer_style(cer: float) -> str:
    if cer < 0.1:
        return "green"
    if cer < 0.3:
        return "yellow"
    return "red"


def _print_summary(results: list[BenchmarkResult], has_refs: bool) -> None:
    console.print()
    table = Table(title="Итоги бенчмарка", show_header=True, header_style="bold magenta")
    table.add_column("Модель", style="cyan", no_wrap=True)
    table.add_column("Загрузка", justify="right")
    table.add_column("Ср. inference", justify="right")
    table.add_column("Ср. RTF", justify="right")
    if has_refs:
        table.add_column("CER норм.", justify="right")
        table.add_column("CER точн.", justify="right")
    table.add_column("Файлов", justify="right")

    for r in results:
        rtf_style = _rtf_style(r.avg_rtf)
        row = [
            r.model_name,
            f"{r.load_time:.2f}s",
            f"{r.avg_inference_time:.3f}s",
            f"[{rtf_style}]{r.avg_rtf:.2f}[/{rtf_style}]",
        ]
        if has_refs:
            if r.avg_cer_norm is not None:
                s = _cer_style(r.avg_cer_norm)
                row.append(f"[{s}]{r.avg_cer_norm:.0%}[/{s}]")
            else:
                row.append("—")
            if r.avg_cer_raw is not None:
                s = _cer_style(r.avg_cer_raw)
                row.append(f"[{s}]{r.avg_cer_raw:.0%}[/{s}]")
            else:
                row.append("—")
        row.append(str(len(r.file_results)))
        table.add_row(*row)

    console.print(table)
    if has_refs:
        console.print("[dim]CER норм. = без пунктуации/регистра  |  CER точн. = с пунктуацией[/dim]")


def run_benchmark(
    models: list[str],
    audio_files: list[Path],
    n_files: int | None = None,
    references: dict[str, str] | None = None,
) -> list[BenchmarkResult]:
    """Запустить бенчмарк распознавания речи для выбранных моделей."""
    files = audio_files[:n_files] if n_files is not None else audio_files
    refs = references or {}
    has_refs = bool(refs)

    if not files:
        console.print("[red]Аудиофайлы не найдены[/red]")
        return []

    ref_note = f", {sum(1 for f in files if f.name in refs)}/{len(files)} с эталонами" if refs else ""
    console.print(
        f"[bold]Бенчмарк: {len(models)} модел(ей) × {len(files)} файл(ов){ref_note}[/bold]"
    )

    results: list[BenchmarkResult] = []
    for model_key in models:
        result = _benchmark_model(model_key, files, refs)
        if result is not None:
            results.append(result)

    if results:
        _print_summary(results, has_refs)

    return results
