import numpy as np

from src.services.kokoro_service import KokoroTTSServiceAdapter


class DummyTTS:
    sample_rate = 24000

    def speak(self, text: str):
        assert text == "你好"
        return np.zeros((2400,), dtype=np.float32)


def test_kokoro_adapter_synthesize():
    adapter = KokoroTTSServiceAdapter(tts=DummyTTS())
    audio, sample_rate = adapter.synthesize_text("你好")
    assert isinstance(audio, np.ndarray)
    assert sample_rate == 24000
