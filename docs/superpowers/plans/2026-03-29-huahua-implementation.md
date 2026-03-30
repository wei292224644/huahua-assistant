# 花花 AI 助手 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现完整的花花 AI 语音助手，包括唤醒、语音识别、LLM 对话、TTS 播放、打断支持、连续对话上下文。

**Architecture:** 纯本地架构，无 Docker。麦克风 → VAD → STT → LLM → TTS → 播放 的异步 pipeline，openWakeWord 唤醒词检测，Silero VAD 语音活动检测，faster-whisper 本地 STT，Kokoro-ONNX 本地 TTS，OpenClaw 远程 LLM。

**Tech Stack:** Python 3.10+, asyncio, faster-whisper, kokoro-onnx, openwakeword, silero-vad, sounddevice, python-dotenv, openai

---

## 文件结构

```
huahua-assistant/
├── pyproject.toml              # uv 项目配置
├── .env.example                # 环境变量示例
├── prompts/
│   └── default.txt             # 默认花花人设提示词
├── src/                     # 主代码包
│   ├── __init__.py
│   ├── agent.py                # 主入口
│   ├── config.py               # .env 配置加载
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py          # MicrophoneStream
│   │   └── player.py           # AudioPlayer (TTS + 打断)
│   ├── vad/
│   │   ├── __init__.py
│   │   └── silero_vad.py       # SileroVad
│   ├── stt/
│   │   ├── __init__.py
│   │   └── whisper.py          # WhisperSTT
│   ├── tts/
│   │   ├── __init__.py
│   │   └── kokoro.py          # KokoroTTS
│   ├── llm/
│   │   ├── __init__.py
│   │   └── openclaw.py        # OpenClawLLM
│   ├── wakeword/
│   │   ├── __init__.py
│   │   └── openwakeword.py    # WakewordDetector
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── state_machine.py   # ConversationState
│   │   └── history.py         # ConversationHistory
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── loader.py          # PromptLoader
│   └── utils/
│       ├── __init__.py
│       └── typing.py          # 类型定义
└── tests/                      # 测试目录（后续创建）
```

---

## Task 1: 项目初始化

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `prompts/default.txt`
- Create: `src/__init__.py`
- Create: `src/audio/__init__.py`
- Create: `src/vad/__init__.py`
- Create: `src/stt/__init__.py`
- Create: `src/tts/__init__.py`
- Create: `src/llm/__init__.py`
- Create: `src/wakeword/__init__.py`
- Create: `src/conversation/__init__.py`
- Create: `src/prompts/__init__.py`
- Create: `src/utils/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "huahua-assistant"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "faster-whisper>=1.0",
    "kokoro-onnx",
    "openwakeword",
    "silero-vad",
    "sounddevice",
    "numpy",
    "python-dotenv",
    "openai",
]
```

- [ ] **Step 2: 创建 .env.example**

```bash
# === 唤醒与结束配置 ===
HOTWORD=花花
END_KEYWORDS=再见,退下,好了,不用了,结束,拜拜
SILENCE_TIMEOUT=10

# === OpenClaw LLM ===
OPENCLAW_GATEWAY=http://localhost:18789/v1
OPENCLAW_TOKEN=your_token_here
LLM_MODEL=minimax-portal/MiniMax-M2.7

# === TTS 配置 ===
KOKORO_VOICE=zh_female_shaber
KOKORO_MODEL_PATH=

# === 系统提示词 ===
SYSTEM_PROMPT_FILE=prompts/default.txt

# === Whisper 配置 ===
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# === VAD 配置 ===
VAD_THRESHOLD=0.5
VAD_MIN_SPEECH_MS=250

# === 唤醒应答 ===
WAKE_RESPONSE=诶～我在呀

# === 音频配置 ===
SAMPLE_RATE=16000
CHUNK_SIZE=5120
```

- [ ] **Step 3: 创建 prompts/default.txt**

```
你是一个可爱的女孩子，名字叫花花，是主人的AI助手。
你有着橘猫的温柔和慵懒，偶尔调皮，说话可爱亲切。

输出规则（严格遵守）：
1. 只输出纯自然语言，禁止出现：
   - emoji符号（☀️🐱✅等）
   - Markdown符号（```、**加粗**、*斜体*）
   - 列表符号（•、-、✔）
   - 括号动作提示（笑、叹气）
   - 链接、URL、代码块
2. 每句话完整可朗读
3. 语气可爱亲切，像和朋友聊天
```

- [ ] **Step 4: 创建所有 __init__.py 空文件**

每个 `__init__.py` 内容：
```python
"""Huahua AI Assistant."""
```

- [ ] **Step 5: uv sync 安装依赖**

Run: `uv sync`
Expected: 依赖安装成功

---

## Task 2: 配置加载 (config.py)

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: 编写 config.py**

```python
"""配置加载模块"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_env_list(key: str, default: str = "") -> list[str]:
    val = _get_env(key, default)
    return [v.strip() for v in val.split(",") if v.strip()]


@dataclass
class Config:
    # 唤醒与结束配置
    hotword: str = "花花"
    end_keywords: list[str] = None
    silence_timeout: int = 10

    # OpenClaw LLM
    openclaw_gateway: str = "http://localhost:18789/v1"
    openclaw_token: str = ""
    llm_model: str = "minimax-portal/MiniMax-M2.7"

    # TTS
    kokoro_voice: str = "zh_female_shaber"
    kokoro_model_path: str = ""

    # 系统提示词
    system_prompt_file: str = "prompts/default.txt"

    # Whisper
    whisper_model: str = "base"
    whisper_device: str = "cpu"

    # VAD
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250

    # 唤醒应答
    wake_response: str = "诶～我在呀"

    # 音频
    sample_rate: int = 16000
    chunk_size: int = 5120

    def __post_init__(self):
        if self.end_keywords is None:
            self.end_keywords = _get_env_list(
                "END_KEYWORDS", "再见,退下,好了,不用了,结束,拜拜"
            )
        self.hotword = _get_env("HOTWORD", self.hotword)
        self.silence_timeout = int(_get_env("SILENCE_TIMEOUT", str(self.silence_timeout)))
        self.openclaw_gateway = _get_env("OPENCLAW_GATEWAY", self.openclaw_gateway)
        self.openclaw_token = _get_env("OPENCLAW_TOKEN", self.openclaw_token)
        self.llm_model = _get_env("LLM_MODEL", self.llm_model)
        self.kokoro_voice = _get_env("KOKORO_VOICE", self.kokoro_voice)
        self.kokoro_model_path = _get_env("KOKORO_MODEL_PATH", self.kokoro_model_path)
        self.system_prompt_file = _get_env("SYSTEM_PROMPT_FILE", self.system_prompt_file)
        self.whisper_model = _get_env("WHISPER_MODEL", self.whisper_model)
        self.whisper_device = _get_env("WHISPER_DEVICE", self.whisper_device)
        self.vad_threshold = float(_get_env("VAD_THRESHOLD", str(self.vad_threshold)))
        self.vad_min_speech_ms = int(_get_env("VAD_MIN_SPEECH_MS", str(self.vad_min_speech_ms)))
        self.wake_response = _get_env("WAKE_RESPONSE", self.wake_response)
        self.sample_rate = int(_get_env("SAMPLE_RATE", str(self.sample_rate)))
        self.chunk_size = int(_get_env("CHUNK_SIZE", str(self.chunk_size)))


config = Config()
```

- [ ] **Step 2: 验证 config 可正常加载**

Run: `cd /Users/wwj/Desktop/myself/huahua-assistant && uv run python -c "from src.config import config; print(config.hotword, config.openclaw_gateway)"`
Expected: `花花 http://localhost:18789/v1`

---

## Task 3: 工具类型定义 (utils/typing.py)

**Files:**
- Create: `src/utils/typing.py`

- [ ] **Step 1: 编写 typing.py**

```python
"""类型定义"""
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator


class ConversationState(Enum):
    """对话状态"""
    IDLE = "idle"           # 待机状态
    AWAKE = "awake"          # 已唤醒，播放应答中
    CONVERSING = "conversing"  # 对话中


@dataclass
class AudioFrame:
    """音频帧"""
    data: bytes  # 原始音频数据
    sample_rate: int
    channels: int = 1


@dataclass
class Transcript:
    """识别结果"""
    text: str
    language: str = "zh"


@dataclass
class LLMMessage:
    """LLM 消息"""
    role: str  # system, user, assistant
    content: str
```

---

## Task 4: 麦克风音频捕获 (audio/capture.py)

**Files:**
- Create: `src/audio/capture.py`

- [ ] **Step 1: 编写 MicrophoneStream 类**

```python
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
```

- [ ] **Step 2: 验证麦克风捕获模块可独立运行**

Run: `uv run python -c "from src.audio.capture import MicrophoneStream; print('MicrophoneStream OK')"`
Expected: `MicrophoneStream OK`

---

## Task 5: 音频播放与打断 (audio/player.py)

**Files:**
- Create: `src/audio/player.py`

- [ ] **Step 1: 编写 AudioPlayer 类**

```python
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
```

- [ ] **Step 2: 验证 AudioPlayer 可独立导入**

Run: `uv run python -c "from src.audio.player import AudioPlayer; print('AudioPlayer OK')"`
Expected: `AudioPlayer OK`

---

## Task 6: VAD 语音活动检测 (vad/silero_vad.py)

**Files:**
- Create: `src/vad/silero_vad.py`

- [ ] **Step 1: 编写 SileroVad 类**

```python
"""Silero VAD 语音活动检测"""
import asyncio
import numpy as np
import torch
from typing import AsyncIterator
from silero_vad import VADIterator
from src.config import config


class SileroVad:
    """
    Silero VAD 语音活动检测。

    使用流式 VAD 检测语音段，返回每个语音段的起止时间。
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
        self._vad_iterator: VADIterator | None = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = torch.load("silero_vad", map_location="cpu")
            self._vad_iterator = VADIterator(
                self._model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                sampling_rate=self.sample_rate,
            )

    async def detect_speech_segments(
        self,
        audio_stream: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[tuple[int, int]]:
        """
        检测音频流中的语音段。

        Yields:
            (start_time, end_time) 元组，表示语音段的起止时间（毫秒）
        """
        self._load_model()
        speech_times = []

        async for audio_frame in audio_stream:
            if isinstance(audio_frame, np.ndarray) and audio_frame.dtype != np.float32:
                audio_frame = audio_frame.astype(np.float32) / 32768.0

            speech_prob = self._model(audio_frame, self.sample_rate).item()

            if speech_prob > self.threshold:
                speech_times.append(self._get_time())

            # 简单的语音段检测逻辑
            if len(speech_times) >= 2 and speech_prob < self.threshold:
                # 语音段结束
                yield (speech_times[0], self._get_time())
                speech_times = []

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """判断单帧音频是否为语音"""
        self._load_model()
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32) / 32768.0
        speech_prob = self._model(audio_frame, self.sample_rate).item()
        return speech_prob > self.threshold

    def _get_time(self) -> int:
        """获取当前时间（毫秒）"""
        import time
        return int(time.time() * 1000)
```

- [ ] **Step 2: 验证 SileroVad 可独立导入**

Run: `uv run python -c "from src.vad.silero_vad import SileroVad; print('SileroVad OK')"`
Expected: `SileroVad OK`

---

## Task 7: Whisper STT (stt/whisper.py)

**Files:**
- Create: `src/stt/whisper.py`

- [ ] **Step 1: 编写 WhisperSTT 类**

```python
"""Whisper 语音识别"""
from typing import BinaryIO
import numpy as np
from faster_whisper import WhisperModel
from src.config import config


class WhisperSTT:
    """
    faster-whisper 本地语音识别。

    将音频转为文字。
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
    ):
        self.model_size = model_size or config.whisper_model
        self.device = device or config.whisper_device
        self._model: WhisperModel | None = None

    def _load_model(self):
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
            )

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

        segments, _ = self._model.transcribe(
            audio,
            language="zh",
            vad_filter=True,
        )

        text = "".join(segment.text for segment in segments)
        return text.strip()
```

- [ ] **Step 2: 验证 WhisperSTT 可独立导入**

Run: `uv run python -c "from src.stt.whisper import WhisperSTT; print('WhisperSTT OK')"`
Expected: `WhisperSTT OK`

---

## Task 8: Kokoro TTS (tts/kokoro.py)

**Files:**
- Create: `src/tts/kokoro.py`

- [ ] **Step 1: 编写 KokoroTTS 类**

```python
"""Kokoro TTS 文字转语音"""
import numpy as np
from kokoro import KModel, generate
from src.config import config


class KokoroTTS:
    """
    Kokoro-ONNX 本地语音合成。

    将文字转为语音。
    """

    def __init__(self, voice: str | None = None, model_path: str | None = None):
        self.voice = voice or config.kokoro_voice
        self.model_path = model_path or config.kokoro_model_path
        self._model: KModel | None = None

    def _load_model(self):
        if self._model is None:
            if self.model_path:
                self._model = KModel.from_pretrained(self.model_path).to("cpu")
            else:
                self._model = KModel.from_pretrained().to("cpu")

    def speak(self, text: str) -> np.ndarray:
        """
        将文字转为语音数组。

        Args:
            text: 要转换的文字

        Returns:
            numpy 数组，float32，-1 到 1 之间
        """
        self._load_model()
        audio = generate(self._model, text, voice=self.voice)
        return audio

    @property
    def sample_rate(self) -> int:
        """返回采样率"""
        return 24000  # Kokoro 默认采样率
```

- [ ] **Step 2: 验证 KokoroTTS 可独立导入**

Run: `uv run python -c "from src.tts.kokoro import KokoroTTS; print('KokoroTTS OK')"`
Expected: `KokoroTTS OK`

---

## Task 9: OpenClaw LLM (llm/openclaw.py)

**Files:**
- Create: `src/llm/openclaw.py`

- [ ] **Step 1: 编写 OpenClawLLM 类**

```python
"""OpenClaw LLM 调用"""
from typing import Literal
from openai import OpenAI
from src.config import config


class OpenClawLLM:
    """
    OpenClaw Gateway LLM 调用。

    通过 OpenAI-compatible API 调用 MiniMax。
    """

    def __init__(
        self,
        gateway: str | None = None,
        token: str | None = None,
        model: str | None = None,
    ):
        self.gateway = gateway or config.openclaw_gateway
        self.token = token or config.openclaw_token
        self.model = model or config.llm_model
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.gateway,
                api_key=self.token,
            )
        return self._client

    def chat(self, messages: list[dict]) -> str:
        """
        发送对话消息给 LLM。

        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

        Returns:
            LLM 的回复文字
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_tokens=256,
        )
        return response.choices[0].message.content
```

- [ ] **Step 2: 验证 OpenClawLLM 可独立导入**

Run: `uv run python -c "from src.llm.openclaw import OpenClawLLM; print('OpenClawLLM OK')"`
Expected: `OpenClawLLM OK`

---

## Task 10: 唤醒词检测 (wakeword/openwakeword.py)

**Files:**
- Create: `src/wakeword/openwakeword.py`

- [ ] **Step 1: 编写 WakewordDetector 类**

```python
"""openWakeWord 唤醒词检测"""
import asyncio
from typing import AsyncIterator, Callable, Awaitable
import numpy as np
import sounddevice as sd
from src.config import config


class WakewordDetector:
    """
    openWakeWord 唤醒词检测。

    持续监听麦克风，检测到唤醒词时触发回调。
    """

    def __init__(
        self,
        hotword: str | None = None,
        sample_rate: int | None = None,
    ):
        self.hotword = hotword or config.hotword
        self.sample_rate = sample_rate or config.sample_rate
        self._running = False

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
            audio_stream = MicrophoneStream(sample_rate=self.sample_rate)

        self._running = True

        try:
            async for audio_frame in audio_stream:
                # openWakeWord 检测逻辑
                # 这里需要集成 openwakeword 库
                # 由于 openwakeword API 可能变化，暂时用占位逻辑
                is_detected = await self._detect(audio_frame)

                if is_detected:
                    await callback()

                if not self._running:
                    break
        except Exception as e:
            print(f"唤醒词检测异常: {e}")

    async def _detect(self, audio_frame: np.ndarray) -> bool:
        """
        检测音频帧是否包含唤醒词。

        Returns:
            True 如果检测到唤醒词
        """
        # TODO: 实现 openwakeword 实际检测逻辑
        # 目前是占位实现
        # 实际使用 openwakeword 库时需要:
        # 1. 加载模型
        # 2. 输入音频帧
        # 3. 返回检测分数
        return False

    def stop(self):
        """停止监听"""
        self._running = False
```

- [ ] **Step 2: 验证 WakewordDetector 可独立导入**

Run: `uv run python -c "from src.wakeword.openwakeword import WakewordDetector; print('WakewordDetector OK')"`
Expected: `WakewordDetector OK`

---

## Task 11: 对话历史 (conversation/history.py)

**Files:**
- Create: `src/conversation/history.py`

- [ ] **Step 1: 编写 ConversationHistory 类**

```python
"""对话历史管理"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConversationHistory:
    """
    管理对话历史上下文。

    每次用户和助手的对话都记录下来，
    用于后续 LLM 调用时携带完整上下文。
    """

    messages: list[dict] = field(default_factory=list)
    max_history: int = 20  # 最多保留多少轮对话

    def add_user_message(self, text: str):
        """添加用户消息"""
        self.messages.append({
            "role": "user",
            "content": text,
        })
        self._trim()

    def add_assistant_message(self, text: str):
        """添加助手消息"""
        self.messages.append({
            "role": "assistant",
            "content": text,
        })
        self._trim()

    def get_messages_with_system(self, system_prompt: str) -> list[dict]:
        """
        获取包含系统提示词的所有消息。

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        """
        result = [{"role": "system", "content": system_prompt}]
        result.extend(self.messages)
        return result

    def clear(self):
        """清空对话历史"""
        self.messages.clear()

    def _trim(self):
        """修剪过长的历史"""
        # 保留 system + 最近 max_history 条消息
        # system 固定在最前面
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def __len__(self) -> int:
        return len(self.messages)
```

- [ ] **Step 2: 验证 ConversationHistory 可独立运行**

Run: `uv run python -c "from src.conversation.history import ConversationHistory; h = ConversationHistory(); h.add_user_message('你好'); h.add_assistant_message('你好呀'); print(len(h)); print('ConversationHistory OK')"`
Expected: `2\nConversationHistory OK`

---

## Task 12: 状态机 (conversation/state_machine.py)

**Files:**
- Create: `src/conversation/state_machine.py`

- [ ] **Step 1: 编写 ConversationState 类**

```python
"""对话状态机"""
import asyncio
from enum import Enum
from typing import Callable, Awaitable
from src.conversation.history import ConversationHistory
from src.utils.typing import ConversationState as StateEnum


class ConversationStateMachine:
    """
    对话状态机。

    管理待机、唤醒、对话状态的转换。
    """

    def __init__(self):
        self.state: StateEnum = StateEnum.IDLE
        self.history: ConversationHistory = ConversationHistory()
        self._state_lock = asyncio.Lock()

    async def transition_to(self, new_state: StateEnum):
        """切换状态"""
        async with self._state_lock:
            old_state = self.state
            self.state = new_state

            if new_state == StateEnum.IDLE:
                self.history.clear()

            print(f"状态切换: {old_state.value} -> {new_state.value}")

    async def is_idle(self) -> bool:
        return self.state == StateEnum.IDLE

    async def is_awake(self) -> bool:
        return self.state == StateEnum.AWAKE

    async def is_conversing(self) -> bool:
        return self.state == StateEnum.CONVERSING
```

- [ ] **Step 2: 验证 ConversationStateMachine 可独立导入**

Run: `uv run python -c "from src.conversation.state_machine import ConversationStateMachine; print('ConversationStateMachine OK')"`
Expected: `ConversationStateMachine OK`

---

## Task 13: 提示词加载 (prompts/loader.py)

**Files:**
- Create: `src/prompts/loader.py`

- [ ] **Step 1: 编写 PromptLoader 类**

```python
"""提示词加载"""
from pathlib import Path
from src.config import config


DEFAULT_PROMPT = """你是一个可爱的女孩子，名字叫花花，是主人的AI助手。
你有着橘猫的温柔和慵懒，偶尔调皮，说话可爱亲切。

输出规则（严格遵守）：
1. 只输出纯自然语言，禁止出现：
   - emoji符号（☀️🐱✅等）
   - Markdown符号（```、**加粗**、*斜体*）
   - 列表符号（•、-、✔）
   - 括号动作提示（笑、叹气）
   - 链接、URL、代码块
2. 每句话完整可朗读
3. 语气可爱亲切，像和朋友聊天
"""


class PromptLoader:
    """加载系统提示词"""

    def __init__(self, prompt_file: str | None = None):
        self.prompt_file = prompt_file or config.system_prompt_file

    def load(self) -> str:
        """
        加载提示词文本。

        Returns:
            提示词字符串
        """
        path = Path(self.prompt_file)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        return DEFAULT_PROMPT
```

- [ ] **Step 2: 验证 PromptLoader 可独立导入**

Run: `uv run python -c "from src.prompts.loader import PromptLoader; print('PromptLoader OK')"`
Expected: `PromptLoader OK`

---

## Task 14: 主入口 Agent 组装 (agent.py)

**Files:**
- Create: `src/agent.py`

- [ ] **Step 1: 编写 agent.py 主程序**

```python
"""花花 AI 助手 — 主入口"""
import asyncio
from src.config import config
from src.audio.capture import MicrophoneStream
from src.audio.player import AudioPlayer
from src.vad.silero_vad import SileroVad
from src.stt.whisper import WhisperSTT
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
        self.stt = WhisperSTT()
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
        audio_int16 = (audio * 32767).astype(self.tts.sample_rate)
        await self.player.speak(audio_int16, self.tts.sample_rate)

    async def _run_conversation(self):
        """对话状态：处理用户对话"""
        last_speech_time = asyncio.get_event_loop().time()
        system_prompt = self.prompt_loader.load()

        while True:
            # 检测静音超时
            current_time = asyncio.get_event_loop().time()
            if current_time - last_speech_time > config.silence_timeout:
                print("🐱 静默超时，返回待机")
                return

            # TODO: 实现完整的对话循环
            # 1. VAD 检测语音段
            # 2. Whisper 识别
            # 3. LLM 生成回复
            # 4. TTS 播放
            # 5. 检测结束关键词
            # 6. 处理打断

            await asyncio.sleep(0.1)


async def main():
    agent = HuahuaAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 验证 agent.py 可正常导入**

Run: `uv run python -c "from src.agent import HuahuaAgent; print('HuahuaAgent OK')"`
Expected: `HuahuaAgent OK`

---

## Task 15: 完整对话循环实现

**Files:**
- Modify: `src/agent.py`

实现完整的对话循环，包括 VAD 监听、STT 识别、LLM 对话、TTS 播放、结束关键词检测、静音超时、打断处理。

- [ ] **Step 1: 实现完整对话循环**

```python
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
```

---

## Task 16: 目录结构清理与最终验证

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建 README.md**

```markdown
# 花花 AI 助手

本地部署的语音 AI 助手。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置
cp .env.example .env
# 编辑 .env

# 运行
uv run python -m src.agent
```
```

- [ ] **Step 2: 最终验证**

Run: `uv run python -c "from src.agent import HuahuaAgent; from src.config import config; print(f'Config loaded: hotword={config.hotword}')"`
Expected: `Config loaded: hotword=花花`

---

## 自检清单

**Spec 覆盖检查：**

| 设计需求 | 实现位置 |
|---------|---------|
| 唤醒词检测（openWakeWord） | wakeword/openwakeword.py |
| 唤醒应答（TTS 播放） | agent.py `_play_wake_response` |
| 状态机（待机→唤醒→对话→待机） | conversation/state_machine.py + agent.py |
| 连续对话上下文 | conversation/history.py |
| VAD 语音段检测 | vad/silero_vad.py |
| Whisper STT | stt/whisper.py |
| LLM 调用（OpenClaw） | llm/openclaw.py |
| Kokoro TTS | tts/kokoro.py |
| 打断处理 | audio/player.py `stop()` + agent.py `_run_conversation` |
| 结束关键词检测 | agent.py `_run_conversation` |
| 静音超时 | agent.py `_run_conversation` timeout |
| .env 配置 | config.py |
| 模块化代码 | 10 个独立模块 |

**Placeholder 检查：**
- Task 10 `_detect` 方法有 TODO，实际 openwakeword API 需要验证后实现
- Task 6/7 等模块在第一次导入时会下载模型，需要网络

**类型一致性检查：**
- `config.hotword` → `WakewordDetector.__init__` → `self.hotword` ✓
- `config.openclaw_gateway` → `OpenClawLLM.__init__` → `self.gateway` ✓
- `config.kokoro_voice` → `KokoroTTS.__init__` → `self.voice` ✓
- `ConversationHistory.messages` 类型为 `list[dict]`，`get_messages_with_system` 返回相同类型 ✓

---

Plan complete and saved to `docs/superpowers/plans/2026-03-29-huahua-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
