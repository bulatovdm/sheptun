# pyright: reportPrivateUsage=false
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheptun.benchmark import (
    BenchmarkResult,
    FileResult,
    _get_wav_duration,
    _load_wav_as_bytes,
    _rtf_style,
    run_benchmark,
)
from sheptun.types import RecognitionResult


def _write_test_wav(path: Path, samples: int = 16000, sample_rate: int = 16000) -> None:
    audio_int16 = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


class TestFileResult:
    def test_rtf_field(self) -> None:
        r = FileResult(filename="a.wav", text="тест", reference=None, inference_time=0.5, audio_duration=2.0, rtf=0.25)
        assert r.rtf == 0.25


class TestBenchmarkResult:
    def test_avg_rtf_empty(self) -> None:
        r = BenchmarkResult(model_name="test", load_time=1.0)
        assert r.avg_rtf == 0.0

    def test_avg_inference_time_empty(self) -> None:
        r = BenchmarkResult(model_name="test", load_time=1.0)
        assert r.avg_inference_time == 0.0

    def test_avg_rtf_with_results(self) -> None:
        r = BenchmarkResult(model_name="test", load_time=1.0)
        r.file_results = [
            FileResult("a.wav", "t", None, 0.4, 2.0, 0.2),
            FileResult("b.wav", "t", None, 0.6, 2.0, 0.3),
        ]
        assert r.avg_rtf == pytest.approx(0.25)

    def test_avg_inference_time_with_results(self) -> None:
        r = BenchmarkResult(model_name="test", load_time=1.0)
        r.file_results = [
            FileResult("a.wav", "t", None, 0.4, 2.0, 0.2),
            FileResult("b.wav", "t", None, 0.6, 2.0, 0.3),
        ]
        assert r.avg_inference_time == pytest.approx(0.5)


class TestGetWavDuration:
    def test_duration_one_second(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path, samples=16000, sample_rate=16000)
        assert _get_wav_duration(wav_path) == pytest.approx(1.0)

    def test_duration_half_second(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path, samples=8000, sample_rate=16000)
        assert _get_wav_duration(wav_path) == pytest.approx(0.5)


class TestLoadWavAsBytes:
    def test_returns_bytes_and_sample_rate(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path, samples=3200, sample_rate=16000)

        raw_bytes, sr = _load_wav_as_bytes(wav_path)

        assert sr == 16000
        assert len(raw_bytes) == 3200 * 2  # int16 = 2 bytes per sample

    def test_preserves_audio_content(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        audio_int16 = np.array([100, -200, 300, -400], dtype=np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        raw_bytes, _ = _load_wav_as_bytes(wav_path)
        loaded = np.frombuffer(raw_bytes, dtype=np.int16)
        np.testing.assert_array_equal(loaded, audio_int16)


class TestRtfStyle:
    def test_green_below_one(self) -> None:
        assert _rtf_style(0.5) == "green"
        assert _rtf_style(0.99) == "green"

    def test_yellow_between_one_and_two(self) -> None:
        assert _rtf_style(1.0) == "yellow"
        assert _rtf_style(1.5) == "yellow"
        assert _rtf_style(1.99) == "yellow"

    def test_red_two_and_above(self) -> None:
        assert _rtf_style(2.0) == "red"
        assert _rtf_style(5.0) == "red"


class TestRunBenchmark:
    def _make_mock_recognizer(self, text: str | None = "тест") -> Any:
        recognizer = MagicMock()
        if text is not None:
            recognizer.recognize.return_value = RecognitionResult(text=text, confidence=1.0)
        else:
            recognizer.recognize.return_value = None
        return recognizer

    def test_returns_empty_list_no_files(self) -> None:
        results = run_benchmark(["mlx"], [], n_files=5)
        assert results == []

    def test_skips_unavailable_model(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path)

        with patch("sheptun.benchmark._load_recognizer", return_value=None):
            results = run_benchmark(["unknown_model"], [wav_path])

        assert results == []

    def test_returns_results_for_available_model(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path)

        mock_recognizer = self._make_mock_recognizer("открой терминал")

        with patch("sheptun.benchmark._load_recognizer", return_value=mock_recognizer):
            results = run_benchmark(["mlx"], [wav_path])

        assert len(results) == 1
        assert results[0].model_name == "MLX Whisper"
        assert len(results[0].file_results) == 1
        assert results[0].file_results[0].text == "открой терминал"

    def test_n_files_limits_files(self, tmp_path: Path) -> None:
        wav_files = []
        for i in range(5):
            p = tmp_path / f"test{i}.wav"
            _write_test_wav(p)
            wav_files.append(p)

        mock_recognizer = self._make_mock_recognizer()

        with patch("sheptun.benchmark._load_recognizer", return_value=mock_recognizer):
            results = run_benchmark(["mlx"], wav_files, n_files=3)

        assert len(results[0].file_results) == 3

    def test_n_files_none_uses_all(self, tmp_path: Path) -> None:
        wav_files = []
        for i in range(4):
            p = tmp_path / f"test{i}.wav"
            _write_test_wav(p)
            wav_files.append(p)

        mock_recognizer = self._make_mock_recognizer()

        with patch("sheptun.benchmark._load_recognizer", return_value=mock_recognizer):
            results = run_benchmark(["mlx"], wav_files, n_files=None)

        assert len(results[0].file_results) == 4

    def test_handles_none_recognition_result(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path)

        mock_recognizer = self._make_mock_recognizer(None)

        with patch("sheptun.benchmark._load_recognizer", return_value=mock_recognizer):
            results = run_benchmark(["mlx"], [wav_path])

        assert results[0].file_results[0].text is None

    def test_rtf_is_calculated(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path, samples=16000)  # 1 second audio

        mock_recognizer = self._make_mock_recognizer()

        with patch("sheptun.benchmark._load_recognizer", return_value=mock_recognizer):
            results = run_benchmark(["mlx"], [wav_path])

        rtf = results[0].file_results[0].rtf
        assert rtf > 0.0
        # inference_time / 1.0 = inference_time, should be small positive number
        assert rtf < 100.0  # sanity check

    def test_multiple_models(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path)

        mock_recognizer = self._make_mock_recognizer()

        with patch("sheptun.benchmark._load_recognizer", return_value=mock_recognizer):
            results = run_benchmark(["mlx", "whisper"], [wav_path])

        assert len(results) == 2
        model_names = {r.model_name for r in results}
        assert "MLX Whisper" in model_names
        assert "Whisper (CPU)" in model_names
