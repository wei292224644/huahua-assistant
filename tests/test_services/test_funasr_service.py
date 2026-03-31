import numpy as np

from src.services.funasr_service import FunASRServiceAdapter


class DummySTT:
    def transcribe(self, audio):
        assert isinstance(audio, np.ndarray)
        return "你好花花"


def test_funasr_adapter_transcribe():
    adapter = FunASRServiceAdapter(stt=DummySTT())
    audio = np.zeros((1600,), dtype=np.float32)
    text = adapter.transcribe_audio(audio)
    assert text == "你好花花"
