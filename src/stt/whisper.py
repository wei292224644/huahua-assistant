"""FunASR 语音识别"""
import logging
import os
import tempfile
import numpy as np
import soundfile as sf
from funasr import AutoModel
from src.config import config

logger = logging.getLogger(__name__)


class FunASRSTT:
    """
    FunASR 本地语音识别。

    将音频转为文字，使用阿里 FunASR 模型。
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or config.funasr_model
        self.device = device or config.funasr_device
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading FunASR model: {self.model_name} on {self.device}")
            self._model = AutoModel(
                model=self.model_name,
                device=self.device,
                disable_update=True,
            )
            logger.info("FunASR model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        将音频转为文字。

        Args:
            audio: numpy 数组，float32，-1 到 1 之间，或 int16

        Returns:
            识别出的文字
        """
        self._load_model()

        # 确保音频是 float32，-1 到 1 之间
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # FunASR 需要写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        try:
            # 写入临时 WAV 文件
            sf.write(temp_path, audio, config.sample_rate)

            logger.debug(f"Transcribing audio from {temp_path}")
            result = self._model.generate(input=temp_path)

            if result and len(result) > 0:
                text = result[0].get('text', '')
            else:
                text = ''

            logger.debug(f"Transcribed text: '{text}'")
            return text.strip()
        finally:
            os.unlink(temp_path)
