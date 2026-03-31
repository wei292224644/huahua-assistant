"""花花 AI 助手 — Pipecat 生命周期入口。"""

import asyncio

import numpy as np
from pipecat.frames.frames import InputAudioRawFrame, InterruptionFrame

from src.audio.capture import MicrophoneStream
from src.audio.player import AudioPlayer
from src.conversation.history import ConversationHistory
from src.config import config
from src.pipecat_app.pipeline import build_pipeline_components
from src.pipecat_app.session import SessionController
from src.prompts.loader import PromptLoader
from src.tts.kokoro import KokoroTTS
from src.vad.silero_vad import SileroVad
from src.wakeword.openwakeword import WakewordDetector


class HuahuaAgent:
    """One-shot wakeword, then fully Pipecat-driven session manager."""
    INTERRUPT_CONFIRM_FRAMES = 3
    INTERRUPT_RMS_THRESHOLD = 0.02

    def __init__(self) -> None:
        self.player = AudioPlayer()
        self.tts = KokoroTTS()
        self.wakeword = WakewordDetector()
        self.vad = SileroVad()
        self.prompt_loader = PromptLoader()

    async def run(self) -> None:
        """Wake once, then keep all follow-up turns in Pipecat loop."""
        await self._wait_for_wakeword()
        await self._play_wake_response()
        while True:
            await self._run_pipeline_session()
            # Pipecat session ended by end keyword or silence timeout.
            # Do not re-arm wakeword; keep conversation fully in Pipecat lifecycle.
            self.player.stop()
            await asyncio.sleep(0.2)

    async def run_once_for_test(self) -> None:
        """Run one full lifecycle for testability."""
        await self._wait_for_wakeword()
        await self._play_wake_response()
        await self._run_pipeline_session()

    async def _wait_for_wakeword(self) -> None:
        event = asyncio.Event()

        async def on_wakeword() -> None:
            # Stop listening immediately once wakeword is hit.
            self.wakeword.stop()
            event.set()

        listen_task = asyncio.create_task(self.wakeword.listen(on_wakeword))
        await event.wait()
        self.wakeword.stop()
        try:
            await asyncio.wait_for(listen_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            listen_task.cancel()

    async def _play_wake_response(self) -> None:
        audio = self.tts.speak(config.wake_response)
        audio_int16 = (audio * 32767).astype(np.int16)
        await self.player.speak(audio_int16, self.tts.sample_rate)

    async def _run_pipeline_session(self) -> None:
        session = SessionController(
            end_keywords=config.end_keywords,
            silence_timeout=config.silence_timeout,
        )
        history = ConversationHistory()
        deps = build_pipeline_components(
            session_controller=session,
            player=self.player,
            history=history,
            system_prompt=self.prompt_loader.load(),
        )
        runner = deps["runner"]
        task = deps["task"]
        runner_task = asyncio.create_task(runner.run(task))

        mic = MicrophoneStream()
        interrupt_speech_count = 0
        try:
            async for audio_frame in mic:
                if session.is_silence_timeout():
                    await task.stop_when_done()
                    break

                # Detect speech while TTS is playing and stop playback early.
                if audio_frame.dtype == np.int16:
                    frame_f32 = audio_frame.astype(np.float32) / 32768.0
                else:
                    frame_f32 = audio_frame.astype(np.float32)

                # When TTS is playing, do not feed mic frames to STT.
                # Only run interruption detection to avoid self-transcription.
                if self.player.is_playing:
                    if self.player.interruptible:
                        rms = float(np.sqrt(np.mean(np.square(frame_f32)))) if frame_f32.size else 0.0
                        if rms >= self.INTERRUPT_RMS_THRESHOLD and self.vad.is_speech(frame_f32):
                            interrupt_speech_count += 1
                            if interrupt_speech_count >= self.INTERRUPT_CONFIRM_FRAMES:
                                self.player.stop()
                                await task.queue_frame(InterruptionFrame())
                                interrupt_speech_count = 0
                        else:
                            interrupt_speech_count = 0
                    else:
                        interrupt_speech_count = 0
                    continue

                interrupt_speech_count = 0

                pcm_i16 = (np.clip(frame_f32, -1.0, 1.0) * 32767).astype(np.int16)
                await task.queue_frame(
                    InputAudioRawFrame(
                        audio=pcm_i16.tobytes(),
                        sample_rate=config.sample_rate,
                        num_channels=1,
                    )
                )
        finally:
            await task.stop_when_done()
            await runner_task


async def main() -> None:
    agent = HuahuaAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
