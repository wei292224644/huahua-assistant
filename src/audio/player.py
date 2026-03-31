"""TTS 音频播放与打断"""
import asyncio
import time
import numpy as np
import sounddevice as sd


class AudioPlayer:
    """
    TTS 播放模块，支持同步播放、异步播放、打断。

    打断逻辑：TTS 播放期间，VAD 检测到新语音时调用 stop() 停止播放。
    播放开始后有短暂保护窗口，防止扬声器声音被麦克风收到误触发打断。
    """

    # 播放开始后多少秒内忽略打断（防止 TTS 回声触发自打断）
    INTERRUPT_GUARD_SECS = 0.6

    def __init__(self):
        self._playback_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._is_playing = False
        self._play_started_at: float = 0.0
        self._lock = asyncio.Lock()

    @staticmethod
    def _to_float32(audio_data: np.ndarray) -> np.ndarray:
        if audio_data.dtype == np.float32:
            return audio_data
        if audio_data.dtype == np.int16:
            return (audio_data.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        return audio_data.astype(np.float32)

    @staticmethod
    def _resample_linear(audio_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or audio_data.size == 0:
            return audio_data
        duration = audio_data.shape[0] / float(src_rate)
        dst_len = max(1, int(duration * dst_rate))
        src_x = np.linspace(0.0, duration, num=audio_data.shape[0], endpoint=False)
        dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, audio_data).astype(np.float32)

    def _play_with_fallback(self, audio_data: np.ndarray, sample_rate: int):
        """Play once, then fallback to default output sample rate on device errors."""
        audio_f32 = self._to_float32(audio_data)
        try:
            sd.play(audio_f32, samplerate=sample_rate)
            sd.wait()
            return
        except Exception:
            pass

        try:
            out_dev = sd.query_devices(sd.default.device[1], "output")
            fallback_rate = int(out_dev["default_samplerate"])
        except Exception:
            fallback_rate = sample_rate
        if fallback_rate <= 0:
            fallback_rate = sample_rate

        audio_rs = self._resample_linear(audio_f32, sample_rate, fallback_rate)
        sd.play(audio_rs, samplerate=fallback_rate)
        sd.wait()

    def _play_sync(self, audio_data: np.ndarray, sample_rate: int):
        """同步播放音频（阻塞直到播放完毕）"""
        self._is_playing = True
        try:
            self._play_with_fallback(audio_data, sample_rate)
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
            self._play_started_at = time.monotonic()

            def _play():
                try:
                    self._play_with_fallback(audio_data, sample_rate)
                except Exception:
                    pass
                finally:
                    self._is_playing = False

            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, _play)
            self._playback_task = asyncio.ensure_future(future)

    @property
    def interruptible(self) -> bool:
        """播放开始超过保护窗口后才允许打断"""
        if not self._is_playing:
            return False
        return (time.monotonic() - self._play_started_at) > self.INTERRUPT_GUARD_SECS

    def stop(self):
        """立即停止当前播放"""
        sd.stop()
        self._is_playing = False
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()

    @property
    def is_playing(self) -> bool:
        return self._is_playing