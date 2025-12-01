import logging
import threading
from typing import Any

import numpy as np

from sheptun.settings import settings
from sheptun.types import RecognitionResult

logger = logging.getLogger("sheptun")

try:
    import AVFoundation  # type: ignore[import-not-found, import-untyped]
    import Foundation  # type: ignore[import-not-found, import-untyped]
    import objc  # type: ignore[import-not-found, import-untyped]
    import Speech  # type: ignore[import-not-found, import-untyped]

    NSObject: Any = getattr(Foundation, "NSObject")  # noqa: B009
    NSLocale: Any = getattr(Foundation, "NSLocale")  # noqa: B009
    SFSpeechRecognizer: Any = getattr(Speech, "SFSpeechRecognizer")  # noqa: B009
    SFSpeechAudioBufferRecognitionRequest: Any = getattr(  # noqa: B009
        Speech, "SFSpeechAudioBufferRecognitionRequest"
    )
    SFSpeechRecognitionResult: Any = getattr(Speech, "SFSpeechRecognitionResult")  # noqa: B009
    AVAudioFormat: Any = getattr(AVFoundation, "AVAudioFormat")  # noqa: B009
    AVAudioPCMBuffer: Any = getattr(AVFoundation, "AVAudioPCMBuffer")  # noqa: B009
except ImportError as e:
    raise ImportError(
        "Apple Speech Framework not available. Install with: pip install pyobjc-framework-Speech"
    ) from e


class RecognitionDelegate(NSObject):  # type: ignore[misc]
    def init(self) -> "RecognitionDelegate":  # type: ignore[misc]
        self = objc.super(RecognitionDelegate, self).init()  # type: ignore[misc]
        if self is None:
            raise RuntimeError("Failed to initialize RecognitionDelegate")
        self.result: RecognitionResult | None = None
        self.error: str | None = None
        self.finished = threading.Event()
        return self  # type: ignore[return-value]

    def speechRecognitionTask_didFinishRecognition_(self, _task: Any, result: Any) -> None:
        if result.isFinal():
            text = result.bestTranscription().formattedString().strip()
            if text:
                segments = result.bestTranscription().segments()
                confidence = self._calculate_confidence(segments)
                self.result = RecognitionResult(text=text, confidence=confidence)
            self.finished.set()

    def speechRecognitionTask_didFinishSuccessfully_(self, _task: Any, success: bool) -> None:
        if not success:
            self.error = "Recognition failed"
        self.finished.set()

    def _calculate_confidence(self, segments: Any) -> float:
        if not segments or len(segments) == 0:
            return 0.0

        total_confidence = 0.0
        count = 0

        for segment in segments:
            if hasattr(segment, "confidence"):
                total_confidence += float(segment.confidence())
                count += 1

        if count == 0:
            return 0.0

        return total_confidence / count


class AppleSpeechRecognizer:
    def __init__(self, locale: str | None = None, on_device: bool = True) -> None:
        self._locale = locale or settings.apple_locale
        self._on_device = on_device
        ns_locale = NSLocale.alloc().initWithLocaleIdentifier_(self._locale)
        self._recognizer = SFSpeechRecognizer.alloc().initWithLocale_(ns_locale)

        if self._recognizer is None:
            raise RuntimeError(f"Speech recognizer not available for locale: {self._locale}")

        if not self._recognizer.isAvailable():
            raise RuntimeError("Speech recognizer is not available")

        if on_device and not self._recognizer.supportsOnDeviceRecognition():
            logger.warning(
                f"On-device recognition not supported for {self._locale}, falling back to server"
            )
            self._on_device = False

        logger.info(
            f"AppleSpeechRecognizer initialized (locale={self._locale}, on_device={self._on_device})"
        )

    def recognize(self, audio_data: bytes, sample_rate: int) -> RecognitionResult | None:
        try:
            audio_array = self._bytes_to_float_array(audio_data, sample_rate)
            if audio_array is None:
                return None

            request = SFSpeechAudioBufferRecognitionRequest.alloc().init()
            if self._on_device:
                request.setRequiresOnDeviceRecognition_(True)

            audio_format = AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
                1, 16000.0, 1, False
            )

            frame_capacity = len(audio_array)
            buffer = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
                audio_format, frame_capacity
            )
            buffer.setFrameLength_(frame_capacity)

            float_channel_data = buffer.floatChannelData()
            channel_0 = float_channel_data[0]
            for i in range(len(audio_array)):
                channel_0[i] = float(audio_array[i])

            request.appendAudioPCMBuffer_(buffer)
            request.endAudio()

            delegate = RecognitionDelegate.alloc().init()
            task = self._recognizer.recognitionTaskWithRequest_delegate_(request, delegate)
            if task is None:
                logger.error("Failed to create recognition task")
                return None

            if not delegate.finished.wait(timeout=10.0):
                logger.error("Recognition timeout")
                task.cancel()
                return None

            if delegate.error:
                logger.error(f"Recognition error: {delegate.error}")
                return None

            result_obj: RecognitionResult | None = delegate.result
            return result_obj

        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None

    def start_warmup(self) -> None:
        pass

    def stop_warmup(self) -> None:
        pass

    def _bytes_to_float_array(
        self, audio_data: bytes, sample_rate: int
    ) -> np.ndarray[Any, Any] | None:
        if len(audio_data) == 0:
            return None

        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if sample_rate != 16000:
            audio_float32 = self._resample(audio_float32, sample_rate, 16000)

        return audio_float32

    def _resample(
        self, audio: np.ndarray[Any, Any], orig_sr: int, target_sr: int
    ) -> np.ndarray[Any, Any]:
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled: np.ndarray[Any, Any] = np.interp(
            indices, np.arange(len(audio)), audio
        ).astype(np.float32)
        return resampled
