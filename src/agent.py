"""花花 AI 助手 — 主入口"""
import asyncio
import numpy as np
from src.config import config
from src.audio.capture import MicrophoneStream
from src.audio.player import AudioPlayer
from src.vad.silero_vad import SileroVad
from src.stt.whisper import FunASRSTT
from src.tts.kokoro import KokoroTTS
from src.llm.openclaw import OpenClawLLM
from src.wakeword.openwakeword import WakewordDetector
from src.conversation.state_machine import ConversationStateMachine
from src.conversation.history import ConversationHistory
from src.prompts.loader import PromptLoader
from src.utils.typing import ConversationState


class HuahuaAgent:
    """花花 AI 助手主类"""

    def __init__(self):
        self.state_machine = ConversationStateMachine()
        self.history = ConversationHistory()
        self.prompt_loader = PromptLoader()

        # 初始化各模块
        self.mic = MicrophoneStream()
        self.player = AudioPlayer()
        self.vad = SileroVad()
        self.stt = FunASRSTT()
        self.tts = KokoroTTS()
        self.llm = OpenClawLLM()
        self.wakeword = WakewordDetector()

    async def run(self):
        """运行花花助手"""
        print("🐱 花花 AI 助手启动...")

        while True:
            await self.state_machine.transition_to(ConversationState.IDLE)
            print("🐱 待机中，等待唤醒...")

            # 待机状态：等待唤醒词
            await self._wait_for_wakeword()

            # 唤醒后播放应答
            await self.state_machine.transition_to(ConversationState.AWAKE)
            await self._play_wake_response()

            # 进入对话状态
            await self.state_machine.transition_to(ConversationState.CONVERSING)
            await self._run_conversation()

    async def _wait_for_wakeword(self):
        """待机状态：等待唤醒词"""
        event = asyncio.Event()

        async def on_wakeword():
            event.set()

        await self.wakeword.listen(on_wakeword, self.mic)
        await event.wait()
        self.wakeword.stop()

    async def _play_wake_response(self):
        """播放唤醒应答"""
        print(f"🐱 说: {config.wake_response}")
        audio = self.tts.speak(config.wake_response)
        audio_int16 = (audio * 32767).astype(np.int16)
        await self.player.speak(audio_int16, self.tts.sample_rate)

    async def _run_conversation(self):
        """对话状态：处理用户对话"""
        last_speech_time = asyncio.get_event_loop().time()
        system_prompt = self.prompt_loader.load()
        audio_buffer = []
        is_recording = False

        # 用于打断检测的停止事件
        stop_event = asyncio.Event()

        async def vad_monitor():
            """VAD 监控协程，检测打断"""
            nonlocal is_recording, audio_buffer, last_speech_time
            async for audio_frame in self.mic:
                # 如果 TTS 正在播放，检测是否被打断
                if self.player.is_playing:
                    if self.vad.is_speech(audio_frame):
                        # 检测到语音，打断 TTS
                        self.player.stop()
                        stop_event.set()

                # 如果正在录音，收集音频
                if is_recording:
                    audio_buffer.append(audio_frame)

        # 启动 VAD 监控协程
        vad_task = asyncio.create_task(vad_monitor())

        try:
            while True:
                # 等待打断事件或超时
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=config.silence_timeout)
                    stop_event.clear()
                except asyncio.TimeoutError:
                    print("🐱 静默超时，返回待机")
                    return

                # 检测结束关键词（从最近的录音中）
                if audio_buffer:
                    audio = np.concatenate(audio_buffer)
                    text = self.stt.transcribe(audio)

                    # 检查结束关键词
                    if any(kw in text for kw in config.end_keywords):
                        print("🐱 听到结束词，对话结束")
                        return

                    # 添加用户消息到历史
                    self.history.add_user_message(text)
                    last_speech_time = asyncio.get_event_loop().time()

                    # 调用 LLM
                    messages = self.history.get_messages_with_system(system_prompt)
                    response = self.llm.chat(messages)

                    # 添加助手回复到历史
                    self.history.add_assistant_message(response)

                    # TTS 播放
                    audio = self.tts.speak(response)
                    audio_int16 = (audio * 32767).astype(np.int16)
                    await self.player.speak_async(audio_int16, self.tts.sample_rate)

                    # 清空录音缓冲区
                    audio_buffer = []
                else:
                    # 没有录音但触发了打断，说明是噪音，继续监听
                    stop_event.clear()

        finally:
            vad_task.cancel()
            try:
                await vad_task
            except asyncio.CancelledError:
                pass


async def main():
    agent = HuahuaAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
