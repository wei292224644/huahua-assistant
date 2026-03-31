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
        判断音频帧是否包含语音。

        支持任意长度输入：按 512 样本（16kHz）分段逐帧判断，
        任意一段检测到语音则返回 True。
        """
        self._load_model()

        if audio_frame.dtype == np.int16:
            audio = audio_frame.astype(np.float32) / 32768.0
        else:
            audio = audio_frame.astype(np.float32)

        audio = audio.flatten()

        frame_size = 512 if self.sample_rate == 16000 else 256

        # 按 frame_size 分段，逐段检测
        for start in range(0, len(audio), frame_size):
            chunk = audio[start:start + frame_size]
            if len(chunk) < frame_size:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)), mode='constant')
            audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.sample_rate).item()
            if speech_prob > self.threshold:
                return True

        return False
