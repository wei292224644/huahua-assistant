"""唤醒词检测 — VAD + FunASR 方案"""
import asyncio
import logging
from typing import AsyncIterator, Callable, Awaitable
import numpy as np
from src.config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class WakewordDetector:
    """
    唤醒词检测。

    使用 VAD 检测语音段，检测到语音时用 FunASR 识别，
    如果识别结果包含唤醒词则触发回调。
    """

    def __init__(
        self,
        hotword: str | None = None,
        sample_rate: int | None = None,
    ):
        self.hotword = hotword or config.hotword
        self.sample_rate = sample_rate or config.sample_rate
        self._running = False

        # 延迟导入，避免循环依赖
        self._vad = None
        self._stt = None

        # VAD 每次需要 512 样本（16kHz）
        self._vad_chunk_size = 512

    def _get_vad(self):
        if self._vad is None:
            from src.vad.silero_vad import SileroVad
            logger.info("Loading SileroVAD...")
            self._vad = SileroVad()
            logger.info("SileroVAD loaded")
        return self._vad

    def _get_stt(self):
        if self._stt is None:
            from src.stt.whisper import FunASRSTT
            logger.info("Loading FunASRSTT...")
            self._stt = FunASRSTT()
            logger.info("FunASRSTT loaded")
        return self._stt

    async def listen(
        self,
        callback: Callable[[], Awaitable[None]],
        audio_stream: AsyncIterator[np.ndarray] | None = None,
    ):
        """
        开始监听唤醒词。

        Args:
            callback: 检测到唤醒词时调用的异步函数
            audio_stream: 可选，音频流，不提供则自己创建
        """
        from src.audio.capture import MicrophoneStream

        if audio_stream is None:
            logger.info(f"Creating MicrophoneStream at {self.sample_rate}Hz")
            audio_stream = MicrophoneStream(sample_rate=self.sample_rate)

        self._running = True
        vad = self._get_vad()
        stt = self._get_stt()

        logger.info(f"Starting wakeword detection, hotword='{self.hotword}'")

        # 录音状态
        audio_buffer = []
        is_recording = False
        silence_chunks = 0
        total_chunks_processed = 0
        # 连续 3 个 chunk 静音认为语音段结束（约 96ms 静音）
        SPEECH_THRESHOLD = 3
        # 最少 5 个 chunk 才认为是有效语音（约 160ms）
        MIN_AUDIO_CHUNKS = 5

        try:
            async for audio_frame in audio_stream:
                if not self._running:
                    logger.info("Wakeword detection stopped")
                    break

                total_chunks_processed += 1

                # 把每帧分成 512 样本的块
                if audio_frame.dtype == np.int16:
                    audio_frame = audio_frame.astype(np.float32) / 32768.0

                num_chunks = len(audio_frame) // self._vad_chunk_size

                for i in range(num_chunks):
                    chunk = audio_frame[i * self._vad_chunk_size:(i + 1) * self._vad_chunk_size]
                    is_speech = vad.is_speech(chunk)

                    if is_speech:
                        # 检测到语音
                        if not is_recording:
                            # 开始录音
                            is_recording = True
                            audio_buffer = []
                            silence_chunks = 0
                            logger.info("🔊 检测到语音，开始录音...")
                        # 持续录音，直到语音段结束
                        audio_buffer.append(chunk)
                        silence_chunks = 0
                        logger.debug(f"Recording: buffer has {len(audio_buffer)} chunks")
                    else:
                        # 静音
                        if is_recording:
                            audio_buffer.append(chunk)
                            silence_chunks += 1
                            logger.debug(f"Silence while recording: {silence_chunks} chunks")

                            # 连续静音超过阈值，语音段结束
                            if silence_chunks >= SPEECH_THRESHOLD and len(audio_buffer) >= MIN_AUDIO_CHUNKS:
                                logger.info(f"语音段结束，识别中... buffer={len(audio_buffer)} chunks")
                                # 识别语音
                                audio = np.concatenate(audio_buffer)
                                logger.debug(f"Concatenated audio shape: {audio.shape}")
                                try:
                                    logger.info("Calling STT...")
                                    text = stt.transcribe(audio)
                                    logger.info(f"🎤 识别到: '{text}'")
                                    # 检查是否包含唤醒词
                                    if self.hotword in text:
                                        logger.info(f"🐱 唤醒词 '{self.hotword}' 检测到！")
                                        await callback()
                                    else:
                                        logger.info(f"识别内容不包含唤醒词 '{self.hotword}'，继续监听")
                                except Exception as e:
                                    logger.error(f"STT 识别异常: {e}", exc_info=True)
                                # 重置
                                audio_buffer = []
                                is_recording = False
                                silence_chunks = 0
                            elif silence_chunks >= SPEECH_THRESHOLD:
                                # 语音太短，丢弃
                                logger.info(f"语音太短丢弃，buffer={len(audio_buffer)} chunks < {MIN_AUDIO_CHUNKS}")
                                audio_buffer = []
                                is_recording = False
                                silence_chunks = 0

                # 每 100 帧打印一次状态
                if total_chunks_processed % 100 == 0:
                    logger.debug(f"Still listening... total_chunks={total_chunks_processed}")

        except Exception as e:
            logger.error(f"唤醒词检测异常: {e}", exc_info=True)

    def stop(self):
        """停止监听"""
        logger.info("Stopping wakeword detection")
        self._running = False
