"""TTS 音频播放与打断"""
import asyncio
import threading
import numpy as np
import sounddevice as sd
from typing import Callable, Awaitable


class AudioPlayer:
    """
    TTS 播放模块，支持同步播放、异步播放、打断。

    打断逻辑：TTS 播放期间，VAD 检测到新语音时调用 stop() 停止播放。
    """

    def __init__(self):
        self._current_stream: sd.OutputStream | None = None
        self._playback_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._is_playing = False
        self._lock = asyncio.Lock()

    def _play_sync(self, audio_data: np.ndarray, sample_rate: int):
        """同步播放音频（阻塞直到播放完毕）"""
        self._is_playing = True
        try:
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
        finally:
            self._is_playing = False

    async def speak(self, audio_data: np.ndarray, sample_rate: int):
        """
        同步播放音频，等待播放完毕。
        用于播放唤醒应答等短音频。
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._play_sync,
            audio_data,
            sample_rate,
        )

    async def speak_async(self, audio_data: np.ndarray, sample_rate: int):
        """
        异步播放音频，不等待。可被 stop() 打断。
        """
        async with self._lock:
            self._stop_event.clear()
            self._is_playing = True

            def _play():
                try:
                    sd.play(audio_data, samplerate=sample_rate)
                    sd.wait()
                except Exception:
                    pass
                finally:
                    self._is_playing = False

            loop = asyncio.get_event_loop()
            self._playback_task = asyncio.create_task(
                loop.run_in_executor(None, _play)
            )

    def stop(self):
        """立即停止当前播放"""
        sd.stop()
        self._is_playing = False
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()

    @property
    def is_playing(self) -> bool:
        return self._is_playing