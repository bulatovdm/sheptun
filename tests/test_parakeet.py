# pyright: reportPrivateUsage=false
import threading
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheptun.parakeet import _CHANNELS, _SAMPLE_RATE, _SAMPLE_WIDTH, ParakeetRecognizer


def _make_recognizer_no_init(
    transcribe_result: Any = None,
    hallucinations: set[str] | None = None,
) -> ParakeetRecognizer:
    """Create ParakeetRecognizer without calling __init__ (no NeMo needed)."""
    recognizer = object.__new__(ParakeetRecognizer)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = transcribe_result or []
    recognizer._model = mock_model
    recognizer._model_id = "test-model"
    recognizer._hallucinations = hallucinations or set()
    recognizer._transcribe_lock = threading.Lock()
    recognizer._warmup_timer = None
    recognizer._warmup_lock = threading.Lock()
    recognizer._warmup_interval = 0.0
    return recognizer


class TestParakeetRecognizerImportError:
    def test_raises_import_error_without_nemo(self) -> None:
        with (
            patch("builtins.__import__", side_effect=ImportError("No module named 'nemo'")),
            pytest.raises(ImportError),
        ):
            recognizer = object.__new__(ParakeetRecognizer)
            ParakeetRecognizer.__init__(recognizer)  # type: ignore[call-arg]


class TestParakeetModelName:
    def test_model_name_property(self) -> None:
        recognizer = _make_recognizer_no_init()
        assert recognizer.model_name == "test-model"

    def test_default_model_id(self) -> None:
        from sheptun.parakeet import _PARAKEET_MODEL_ID

        assert _PARAKEET_MODEL_ID == "nvidia/parakeet-tdt-0.6b-v3"


class TestWriteTempWav:
    def test_creates_valid_wav_file(self) -> None:
        recognizer = _make_recognizer_no_init()

        audio = np.zeros(1600, dtype=np.float32)
        tmp_wav = recognizer._write_temp_wav(audio)

        try:
            assert tmp_wav.exists()
            assert tmp_wav.suffix == ".wav"
            with wave.open(str(tmp_wav)) as wf:
                assert wf.getnchannels() == _CHANNELS
                assert wf.getsampwidth() == _SAMPLE_WIDTH
                assert wf.getframerate() == _SAMPLE_RATE
                assert wf.getnframes() == 1600
        finally:
            tmp_wav.unlink(missing_ok=True)

    def test_wav_data_is_int16_scaled(self) -> None:
        recognizer = _make_recognizer_no_init()

        audio = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        tmp_wav = recognizer._write_temp_wav(audio)

        try:
            with wave.open(str(tmp_wav)) as wf:
                raw = wf.readframes(4)
            samples = np.frombuffer(raw, dtype=np.int16)
            assert samples[0] == int(0.5 * 32767)
            assert samples[1] == int(-0.5 * 32767)
        finally:
            tmp_wav.unlink(missing_ok=True)

    def test_temp_file_is_in_system_tmpdir(self) -> None:
        import tempfile

        recognizer = _make_recognizer_no_init()
        audio = np.zeros(100, dtype=np.float32)
        tmp_wav = recognizer._write_temp_wav(audio)

        try:
            assert str(tmp_wav).startswith(tempfile.gettempdir())
        finally:
            tmp_wav.unlink(missing_ok=True)


class TestExtractText:
    def test_extracts_text_from_object_with_text_attr(self) -> None:
        recognizer = _make_recognizer_no_init()

        result = MagicMock()
        result.text = "  привет мир  "
        assert recognizer._extract_text([result]) == "привет мир"

    def test_extracts_text_from_plain_string(self) -> None:
        recognizer = _make_recognizer_no_init()
        assert recognizer._extract_text(["  открой терминал  "]) == "открой терминал"

    def test_returns_empty_for_empty_list(self) -> None:
        recognizer = _make_recognizer_no_init()
        assert recognizer._extract_text([]) == ""


class TestParakeetRecognize:
    def _make_audio_bytes(self, samples: int = 3200) -> tuple[bytes, int]:
        audio = np.zeros(samples, dtype=np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes(), 16000

    def test_returns_recognition_result_on_success(self) -> None:
        result = MagicMock()
        result.text = "открой терминал"

        recognizer = _make_recognizer_no_init(transcribe_result=[result])
        audio_bytes, sr = self._make_audio_bytes()
        out = recognizer.recognize(audio_bytes, sr)

        assert out is not None
        assert out.text == "открой терминал"
        assert out.confidence == 1.0

    def test_returns_none_for_whitespace_text(self) -> None:
        result = MagicMock()
        result.text = "   "

        recognizer = _make_recognizer_no_init(transcribe_result=[result])
        audio_bytes, sr = self._make_audio_bytes()
        assert recognizer.recognize(audio_bytes, sr) is None

    def test_returns_none_for_empty_results(self) -> None:
        recognizer = _make_recognizer_no_init(transcribe_result=[])
        audio_bytes, sr = self._make_audio_bytes()
        assert recognizer.recognize(audio_bytes, sr) is None

    def test_filters_hallucinations(self) -> None:
        result = MagicMock()
        result.text = "Продолжение следует..."

        recognizer = _make_recognizer_no_init(
            transcribe_result=[result],
            hallucinations={"продолжение следует..."},
        )
        audio_bytes, sr = self._make_audio_bytes()
        assert recognizer.recognize(audio_bytes, sr) is None

    def test_cleans_up_temp_file_on_success(self) -> None:
        result = MagicMock()
        result.text = "тест"

        recognizer = _make_recognizer_no_init(transcribe_result=[result])
        audio_bytes, sr = self._make_audio_bytes()

        created_paths: list[Path] = []
        original_write = recognizer._write_temp_wav

        def tracking_write(audio_array: Any) -> Path:
            p = original_write(audio_array)
            created_paths.append(p)
            return p

        recognizer._write_temp_wav = tracking_write
        recognizer.recognize(audio_bytes, sr)

        for p in created_paths:
            assert not p.exists(), f"Temp file not cleaned up: {p}"

    def test_cleans_up_temp_file_on_error(self) -> None:
        recognizer = _make_recognizer_no_init()
        recognizer._model.transcribe.side_effect = RuntimeError("model error")

        audio_bytes = np.zeros(3200, dtype=np.int16).tobytes()

        created_paths: list[Path] = []
        original_write = recognizer._write_temp_wav

        def tracking_write(audio_array: Any) -> Path:
            p = original_write(audio_array)
            created_paths.append(p)
            return p

        recognizer._write_temp_wav = tracking_write

        with pytest.raises(RuntimeError):
            recognizer.recognize(audio_bytes, 16000)

        for p in created_paths:
            assert not p.exists(), f"Temp file not cleaned up: {p}"

    def test_transcribe_called_with_wav_path(self) -> None:
        result = MagicMock()
        result.text = "тест"

        recognizer = _make_recognizer_no_init(transcribe_result=[result])
        audio_bytes, sr = self._make_audio_bytes()
        recognizer.recognize(audio_bytes, sr)

        call_args = recognizer._model.transcribe.call_args
        assert call_args is not None
        paths_arg = call_args[0][0]
        assert isinstance(paths_arg, list)
        assert len(paths_arg) == 1
        assert paths_arg[0].endswith(".wav")
