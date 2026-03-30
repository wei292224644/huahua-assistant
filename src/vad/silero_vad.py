"""Silero VAD 语音活动检测"""
import asyncio
import numpy as np
import torch
from typing import AsyncIterator
from silero_vad import VADIterator, load_silero_vad, get_speech_timestamps
from src.config import config


class SileroVad:
    """
    Silero VAD 语音活动检测。

    使用流式 VAD 检测语音段。
    """

    def __init__(
        self,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
        sample_rate: int | None = None,
    ):
        self.threshold = threshold or config.vad_threshold
        self.min_speech_ms = min_speech_ms or config.vad_min_speech_ms
        self.sample_rate = sample_rate or config.sample_rate
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = load_silero_vad()

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """
        判断单帧音频是否为语音。

        audio_frame 应该是 512 样本（16kHz）或 256 样本（8kHz）。
        如果不是，会自动调整。
        """
        self._load_model()

        # 转换为 torch tensor
        if audio_frame.dtype == np.int16:
            audio = audio_frame.astype(np.float32) / 32768.0
        else:
            audio = audio_frame.astype(np.float32)

        # 确保是 1D
        audio = audio.flatten()

        # 调整到正确的样本数
        expected_samples = 512 if self.sample_rate == 16000 else 256

        if len(audio) < expected_samples:
            # 填充到正确长度
            audio = np.pad(audio, (0, expected_samples - len(audio)), mode='constant')
        elif len(audio) > expected_samples:
            # 截断
            audio = audio[:expected_samples]

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # 添加 batch 维度

        with torch.no_grad():
            speech_prob = self._model(audio_tensor, self.sample_rate).item()

        return speech_prob > self.threshold
