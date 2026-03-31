"""Pipecat pipeline builder with local STT/LLM/TTS processors."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from pipecat.frames.frames import EndFrame, InputAudioRawFrame, TTSAudioRawFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.audio.player import AudioPlayer
from src.conversation.history import ConversationHistory
from src.pipecat_app.session import SessionController
from src.services.funasr_service import FunASRServiceAdapter
from src.services.kokoro_service import KokoroTTSServiceAdapter
from src.services.openclaw_service import OpenClawLLMServiceAdapter
from src.vad.silero_vad import SileroVad


def should_terminate(session_controller: SessionController, user_text: str) -> bool:
    """Return True when session should terminate by text or silence timeout."""
    return session_controller.should_end_by_text(user_text) or session_controller.is_silence_timeout()


class LocalSTTProcessor(FrameProcessor):
    """Aggregate speech chunks and emit text frames via local STT adapter."""

    SILENCE_CHUNKS = 5
    MIN_SPEECH_CHUNKS = 1

    def __init__(
        self,
        *,
        stt_adapter: FunASRServiceAdapter,
        vad: SileroVad,
        session_controller: SessionController,
    ) -> None:
        super().__init__(name="LocalSTTProcessor")
        self._stt_adapter = stt_adapter
        self._vad = vad
        self._session_controller = session_controller
        self._buffer: list[np.ndarray] = []
        self._is_recording = False
        self._silence_count = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        # Input audio is 16-bit PCM bytes.
        audio_int16 = np.frombuffer(frame.audio, dtype=np.int16)
        frame_f32 = audio_int16.astype(np.float32) / 32768.0
        is_speech = self._vad.is_speech(frame_f32)

        if is_speech:
            if not self._is_recording:
                self._is_recording = True
                self._buffer = []
                self._silence_count = 0
            self._buffer.append(frame_f32)
            self._silence_count = 0
            return

        if self._is_recording:
            self._buffer.append(frame_f32)
            self._silence_count += 1
            if self._silence_count >= self.SILENCE_CHUNKS:
                if len(self._buffer) >= self.MIN_SPEECH_CHUNKS:
                    audio = np.concatenate(self._buffer)
                    text = self._stt_adapter.transcribe_audio(audio).strip()
                    self._session_controller.mark_user_activity()
                    if text:
                        await self.push_frame(TextFrame(text), FrameDirection.DOWNSTREAM)
                self._buffer = []
                self._is_recording = False
                self._silence_count = 0


class LocalLLMProcessor(FrameProcessor):
    """Convert user text to assistant text with local LLM adapter."""

    def __init__(
        self,
        *,
        llm_adapter: OpenClawLLMServiceAdapter,
        session_controller: SessionController,
        history: ConversationHistory,
        system_prompt: str,
    ) -> None:
        super().__init__(name="LocalLLMProcessor")
        self._llm_adapter = llm_adapter
        self._session_controller = session_controller
        self._history = history
        self._system_prompt = system_prompt

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        user_text = frame.text.strip()
        if should_terminate(self._session_controller, user_text):
            self._history.clear()
            await self.push_frame(EndFrame(reason="session_terminated"), FrameDirection.DOWNSTREAM)
            return

        self._history.add_user_message(user_text)
        messages = self._history.get_messages_with_system(self._system_prompt)
        reply = self._llm_adapter.chat_messages(messages).strip()
        if not reply:
            return
        self._history.add_assistant_message(reply)
        await self.push_frame(TextFrame(reply), FrameDirection.DOWNSTREAM)


class LocalTTSProcessor(FrameProcessor):
    """Convert assistant text to TTSAudioRawFrame."""

    def __init__(self, *, tts_adapter: KokoroTTSServiceAdapter) -> None:
        super().__init__(name="LocalTTSProcessor")
        self._tts_adapter = tts_adapter

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        audio_f32, sample_rate = self._tts_adapter.synthesize_text(frame.text)
        if audio_f32.size == 0:
            return
        audio_i16 = np.clip(audio_f32, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767).astype(np.int16)
        await self.push_frame(
            TTSAudioRawFrame(audio=audio_i16.tobytes(), sample_rate=sample_rate, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )


class LocalPlaybackProcessor(FrameProcessor):
    """Play synthesized audio through existing AudioPlayer."""

    def __init__(self, *, player: AudioPlayer) -> None:
        super().__init__(name="LocalPlaybackProcessor")
        self._player = player

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            audio = np.frombuffer(frame.audio, dtype=np.int16)
            await self._player.speak_async(audio, frame.sample_rate)
            return
        if isinstance(frame, EndFrame):
            self._player.stop()
        await self.push_frame(frame, direction)


def build_pipeline_components(
    session_controller: SessionController,
    *,
    stt_adapter: FunASRServiceAdapter | None = None,
    llm_adapter: OpenClawLLMServiceAdapter | None = None,
    tts_adapter: KokoroTTSServiceAdapter | None = None,
    history: ConversationHistory | None = None,
    system_prompt: str = "",
    player: AudioPlayer | None = None,
) -> dict[str, Any]:
    """Build pipeline/task/runner components with local processors."""
    stt_adapter = stt_adapter or FunASRServiceAdapter()
    llm_adapter = llm_adapter or OpenClawLLMServiceAdapter()
    tts_adapter = tts_adapter or KokoroTTSServiceAdapter()
    history = history or ConversationHistory()
    player = player or AudioPlayer()
    vad = SileroVad()

    stt_processor = LocalSTTProcessor(
        stt_adapter=stt_adapter,
        vad=vad,
        session_controller=session_controller,
    )
    llm_processor = LocalLLMProcessor(
        llm_adapter=llm_adapter,
        session_controller=session_controller,
        history=history,
        system_prompt=system_prompt,
    )
    tts_processor = LocalTTSProcessor(tts_adapter=tts_adapter)
    playback_processor = LocalPlaybackProcessor(player=player)

    pipeline = Pipeline([stt_processor, llm_processor, tts_processor, playback_processor])
    # Use our own SessionController timeout logic in agent loop; avoid double timeout races.
    task = PipelineTask(pipeline, idle_timeout_secs=None)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    runner = PipelineRunner(loop=loop, handle_sigint=False, handle_sigterm=False)

    return {
        "runner": runner,
        "task": task,
        "pipeline": pipeline,
        "stt_adapter": stt_adapter,
        "llm_adapter": llm_adapter,
        "tts_adapter": tts_adapter,
        "session_controller": session_controller,
        "player": player,
        "history": history,
        "vad": vad,
    }
