"""Kokoro TTS 文字转语音"""
import numpy as np
from kokoro import KPipeline
from src.config import config


class KokoroTTS:
    """
    Kokoro-ONNX 本地语音合成。

    将文字转为语音。
    """

    def __init__(self, voice: str | None = None, model_path: str | None = None):
        self.voice = voice or config.kokoro_voice
        self.model_path = model_path or config.kokoro_model_path
        self._pipeline: KPipeline | None = None
        self._lang_code = "z"  # default to Chinese

    def _load_model(self):
        if self._pipeline is None:
            repo_id = self.model_path if self.model_path else None
            self._pipeline = KPipeline(lang_code=self._lang_code, repo_id=repo_id)

    def speak(self, text: str) -> np.ndarray:
        """
        将文字转为语音数组。

        Args:
            text: 要转换的文字

        Returns:
            numpy 数组，float32，-1 到 1 之间
        """
        self._load_model()
        # KPipeline returns a generator of KPipeline.Result objects
        # Result has an audio property that returns torch.FloatTensor or None
        audio_chunks = []
        for result in self._pipeline(text, voice=self.voice):
            audio = result.audio
            if audio is not None:
                # Convert torch tensor to numpy
                audio = audio.cpu().numpy()
                audio_chunks.append(audio)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        # Concatenate all audio chunks
        audio = np.concatenate(audio_chunks, axis=0)
        return audio

    @property
    def sample_rate(self) -> int:
        """返回采样率"""
        return 24000  # Kokoro 默认采样率
