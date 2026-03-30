"""麦克风音频捕获"""
import asyncio
from typing import AsyncIterator
import numpy as np
import sounddevice as sd
from src.config import config


class MicrophoneStream:
    """持续读取麦克风音频，支持异步迭代"""

    def __init__(
        self,
        sample_rate: int | None = None,
        chunk_size: int | None = None,
    ):
        self.sample_rate = sample_rate or config.sample_rate
        self.chunk_size = chunk_size or config.chunk_size
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._closed = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"麦克风警告: {status}")
        self._queue.put_nowait(indata[:, 0].copy())  # 单声道

    async def start(self):
        """启动音频流"""
        loop = asyncio.get_event_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    async def stop(self):
        """停止音频流"""
        self._closed = True
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        """异步迭代音频帧"""
        await self.start()
        try:
            while not self._closed:
                try:
                    audio = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                    yield audio
                except asyncio.TimeoutError:
                    continue
        finally:
            await self.stop()
