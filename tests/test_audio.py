# pyright: reportPrivateUsage=false
import numpy as np

from sheptun.audio import AudioConfig, EnergyVAD, VoiceActivityConfig


class TestAudioConfig:
    def test_default_values(self) -> None:
        config = AudioConfig()

        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.dtype == "int16"
        assert config.blocksize == 1024

    def test_custom_values(self) -> None:
        config = AudioConfig(
            sample_rate=44100,
            channels=2,
            dtype="float32",
            blocksize=2048,
        )

        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.dtype == "float32"
        assert config.blocksize == 2048


class TestVoiceActivityConfig:
    def test_default_values(self) -> None:
        config = VoiceActivityConfig()

        assert config.energy_threshold == 0.01
        assert config.silence_duration == 0.5
        assert config.min_speech_duration == 0.2
        assert config.max_speech_duration == 30.0

    def test_custom_values(self) -> None:
        config = VoiceActivityConfig(
            energy_threshold=0.05,
            silence_duration=1.0,
            min_speech_duration=0.5,
            max_speech_duration=60.0,
        )

        assert config.energy_threshold == 0.05
        assert config.silence_duration == 1.0
        assert config.min_speech_duration == 0.5
        assert config.max_speech_duration == 60.0


class TestEnergyVAD:
    def test_initial_state(self) -> None:
        config = VoiceActivityConfig()
        vad = EnergyVAD(config)

        assert vad.is_speaking is False
        assert vad._silence_samples == 0
        assert vad._speech_samples == 0

    def test_reset_clears_state(self) -> None:
        config = VoiceActivityConfig()
        vad = EnergyVAD(config)

        # Simulate some state
        vad.is_speaking = True
        vad._silence_samples = 100
        vad._speech_samples = 200

        vad.reset()

        assert vad.is_speaking is False
        assert vad._silence_samples == 0
        assert vad._speech_samples == 0

    def test_silence_returns_false(self) -> None:
        config = VoiceActivityConfig(energy_threshold=0.01)
        vad = EnergyVAD(config)

        # Create silent audio (very low amplitude)
        silent_audio = np.zeros(1024, dtype=np.int16).tobytes()

        result = vad.process_chunk(silent_audio, 16000)

        assert result is False

    def test_speech_detected(self) -> None:
        config = VoiceActivityConfig(
            energy_threshold=0.01,
            min_speech_duration=0.1,
            silence_duration=0.1,
        )
        vad = EnergyVAD(config)

        # Create loud audio (high amplitude)
        loud_audio = (np.ones(3200, dtype=np.int16) * 10000).tobytes()

        # Process speech
        vad.process_chunk(loud_audio, 16000)

        assert vad.is_speaking is True

    def test_speech_ends_after_silence(self) -> None:
        config = VoiceActivityConfig(
            energy_threshold=0.01,
            min_speech_duration=0.1,
            silence_duration=0.1,
        )
        vad = EnergyVAD(config)

        # Create loud audio
        loud_audio = (np.ones(3200, dtype=np.int16) * 10000).tobytes()
        # Create silent audio
        silent_audio = np.zeros(3200, dtype=np.int16).tobytes()

        # Process speech then silence
        vad.process_chunk(loud_audio, 16000)  # 0.2 sec of speech
        result = vad.process_chunk(silent_audio, 16000)  # 0.2 sec of silence

        # Should return True because we have enough speech and silence
        assert result is True

    def test_max_speech_duration_triggers_end(self) -> None:
        config = VoiceActivityConfig(
            energy_threshold=0.001,
            max_speech_duration=0.1,  # Very short max
        )
        vad = EnergyVAD(config)

        # Create loud audio longer than max duration
        loud_audio = (np.ones(3200, dtype=np.int16) * 10000).tobytes()

        # Process enough to exceed max
        result = vad.process_chunk(loud_audio, 16000)

        assert result is True

    def test_short_speech_not_detected(self) -> None:
        config = VoiceActivityConfig(
            energy_threshold=0.01,
            min_speech_duration=1.0,  # Require 1 second of speech
            silence_duration=0.1,
        )
        vad = EnergyVAD(config)

        # Create short loud audio (less than min_speech_duration)
        loud_audio = (np.ones(1600, dtype=np.int16) * 10000).tobytes()  # 0.1 sec
        silent_audio = np.zeros(3200, dtype=np.int16).tobytes()  # 0.2 sec

        vad.process_chunk(loud_audio, 16000)
        result = vad.process_chunk(silent_audio, 16000)

        # Should return False because speech was too short
        assert result is False
