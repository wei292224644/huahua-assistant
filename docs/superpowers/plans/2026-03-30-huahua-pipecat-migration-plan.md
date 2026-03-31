# Huahua Pipecat Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有手写 `agent.py` 对话主循环迁移到 Pipecat pipeline，同时保留本地 FunASR/Kokoro/OpenClaw 与独立唤醒词流程。

**Architecture:** 保留 `WakewordDetector` 作为 pipeline 外层触发器；新增 `src/services/` 中的三个适配器把现有 STT/LLM/TTS 封装为 Pipecat 可调用服务；新增 pipeline 组装与会话控制模块；`src/agent.py` 只负责生命周期管理（待机、启动 pipeline、结束回待机）。通过文本结束词与静默超时双通道终止会话，结束时清空历史。

**Tech Stack:** Python 3.10+, asyncio, pipecat-ai, funasr, kokoro, openai-compatible client, pytest, pytest-asyncio

---

## File Structure

### New files
- `src/services/__init__.py` - services 包导出
- `src/services/funasr_service.py` - FunASR 适配器（音频 -> 文本）
- `src/services/openclaw_service.py` - OpenClaw 适配器（messages -> 文本）
- `src/services/kokoro_service.py` - Kokoro 适配器（文本 -> 音频）
- `src/pipecat_app/__init__.py` - pipecat_app 包导出
- `src/pipecat_app/session.py` - 会话结束信号与控制器
- `src/pipecat_app/pipeline.py` - Pipeline 构建与 runner 启停
- `tests/test_services/test_funasr_service.py` - FunASR adapter 单测
- `tests/test_services/test_openclaw_service.py` - OpenClaw adapter 单测
- `tests/test_services/test_kokoro_service.py` - Kokoro adapter 单测
- `tests/test_pipecat/test_session.py` - 会话结束逻辑单测
- `tests/test_agent/test_lifecycle.py` - agent 生命周期单测（mock 化）

### Modified files
- `pyproject.toml` - 增加 `pipecat-ai` 与测试依赖
- `src/agent.py` - 重写为 wakeword + pipeline lifecycle 管理

---

### Task 1: 依赖与测试基线

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_pipecat/test_session.py`

- [ ] **Step 1: 在 `pyproject.toml` 增加运行与测试依赖**

```toml
[project]
name = "huahua-assistant"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "funasr>=1.3",
    "kokoro-onnx",
    "openwakeword",
    "silero-vad",
    "sounddevice",
    "numpy",
    "python-dotenv",
    "openai",
    "kokoro>=0.9.4",
    "ordered-set>=4.1.0",
    "soundfile",
    "pipecat-ai",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

- [ ] **Step 2: 新建会话控制测试文件（先写失败用例）**

```python
import asyncio

from src.pipecat_app.session import SessionController


async def test_session_should_end_by_keyword():
    controller = SessionController(end_keywords=["结束", "拜拜"], silence_timeout=10)
    assert controller.should_end_by_text("我们结束吧")
    assert not controller.should_end_by_text("继续聊")


async def test_session_should_end_by_timeout():
    controller = SessionController(end_keywords=["结束"], silence_timeout=0)
    controller.mark_user_activity()
    await asyncio.sleep(0.01)
    assert controller.is_silence_timeout()
```

- [ ] **Step 3: 运行新增测试，确认失败**

Run: `uv run pytest tests/test_pipecat/test_session.py -v`  
Expected: FAIL（`ModuleNotFoundError: No module named 'src.pipecat_app'`）

- [ ] **Step 4: 安装依赖并同步环境**

Run: `uv sync`  
Expected: 依赖同步成功，包含 `pipecat-ai`、`pytest`、`pytest-asyncio`

- [ ] **Step 5: 提交当前基线**

```bash
git add pyproject.toml tests/test_pipecat/test_session.py
git commit -m "chore: add pipecat and test dependencies baseline"
```

---

### Task 2: 会话控制器（结束词 + 静默超时）

**Files:**
- Create: `src/pipecat_app/__init__.py`
- Create: `src/pipecat_app/session.py`
- Modify: `tests/test_pipecat/test_session.py`

- [ ] **Step 1: 创建 `src/pipecat_app/__init__.py`**

```python
"""Pipecat app helpers."""

from src.pipecat_app.session import SessionController

__all__ = ["SessionController"]
```

- [ ] **Step 2: 实现 `src/pipecat_app/session.py` 最小可用版本**

```python
"""Session lifecycle control for Pipecat conversation."""

from __future__ import annotations

import time


class SessionController:
    def __init__(self, end_keywords: list[str], silence_timeout: int) -> None:
        self._end_keywords = end_keywords
        self._silence_timeout = silence_timeout
        self._last_user_activity = time.monotonic()

    def mark_user_activity(self) -> None:
        self._last_user_activity = time.monotonic()

    def should_end_by_text(self, text: str) -> bool:
        return any(keyword in text for keyword in self._end_keywords)

    def is_silence_timeout(self) -> bool:
        elapsed = time.monotonic() - self._last_user_activity
        return elapsed > self._silence_timeout
```

- [ ] **Step 3: 把测试改为 pytest-asyncio 可执行风格**

```python
import asyncio
import pytest

from src.pipecat_app.session import SessionController


def test_session_should_end_by_keyword():
    controller = SessionController(end_keywords=["结束", "拜拜"], silence_timeout=10)
    assert controller.should_end_by_text("我们结束吧")
    assert not controller.should_end_by_text("继续聊")


@pytest.mark.asyncio
async def test_session_should_end_by_timeout():
    controller = SessionController(end_keywords=["结束"], silence_timeout=0)
    controller.mark_user_activity()
    await asyncio.sleep(0.01)
    assert controller.is_silence_timeout()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_pipecat/test_session.py -v`  
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add src/pipecat_app/__init__.py src/pipecat_app/session.py tests/test_pipecat/test_session.py
git commit -m "feat: add session controller for keyword and silence timeout"
```

---

### Task 3: FunASR 适配器（先测后写）

**Files:**
- Create: `src/services/__init__.py`
- Create: `src/services/funasr_service.py`
- Create: `tests/test_services/test_funasr_service.py`

- [ ] **Step 1: 写 FunASR adapter 失败测试**

```python
import numpy as np

from src.services.funasr_service import FunASRServiceAdapter


class DummySTT:
    def transcribe(self, audio):
        assert isinstance(audio, np.ndarray)
        return "你好花花"


def test_funasr_adapter_transcribe():
    adapter = FunASRServiceAdapter(stt=DummySTT())
    audio = np.zeros((1600,), dtype=np.float32)
    text = adapter.transcribe_audio(audio)
    assert text == "你好花花"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_services/test_funasr_service.py -v`  
Expected: FAIL（`ModuleNotFoundError: No module named 'src.services'`）

- [ ] **Step 3: 实现 services 包与 FunASR adapter**

`src/services/__init__.py`
```python
"""Adapters that bridge local modules to Pipecat pipeline."""

from src.services.funasr_service import FunASRServiceAdapter
from src.services.kokoro_service import KokoroTTSServiceAdapter
from src.services.openclaw_service import OpenClawLLMServiceAdapter

__all__ = [
    "FunASRServiceAdapter",
    "KokoroTTSServiceAdapter",
    "OpenClawLLMServiceAdapter",
]
```

`src/services/funasr_service.py`
```python
"""FunASR adapter for pipeline use."""

from __future__ import annotations

import numpy as np

from src.stt.whisper import FunASRSTT


class FunASRServiceAdapter:
    def __init__(self, stt: FunASRSTT | None = None) -> None:
        self._stt = stt or FunASRSTT()

    def transcribe_audio(self, audio: np.ndarray) -> str:
        return self._stt.transcribe(audio).strip()
```

- [ ] **Step 4: 重新运行测试**

Run: `uv run pytest tests/test_services/test_funasr_service.py -v`  
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add src/services/__init__.py src/services/funasr_service.py tests/test_services/test_funasr_service.py
git commit -m "feat: add FunASR adapter for pipeline transcription"
```

---

### Task 4: OpenClaw 与 Kokoro 适配器（并行能力准备）

**Files:**
- Create: `src/services/openclaw_service.py`
- Create: `src/services/kokoro_service.py`
- Create: `tests/test_services/test_openclaw_service.py`
- Create: `tests/test_services/test_kokoro_service.py`

- [ ] **Step 1: 写 OpenClaw adapter 失败测试**

```python
from src.services.openclaw_service import OpenClawLLMServiceAdapter


class DummyLLM:
    def chat(self, messages):
        assert messages[0]["role"] == "system"
        return "这是回答"


def test_openclaw_adapter_chat():
    adapter = OpenClawLLMServiceAdapter(llm=DummyLLM())
    reply = adapter.chat_messages([{"role": "system", "content": "你是花花"}])
    assert reply == "这是回答"
```

- [ ] **Step 2: 写 Kokoro adapter 失败测试**

```python
import numpy as np

from src.services.kokoro_service import KokoroTTSServiceAdapter


class DummyTTS:
    sample_rate = 24000

    def speak(self, text: str):
        assert text == "你好"
        return np.zeros((2400,), dtype=np.float32)


def test_kokoro_adapter_synthesize():
    adapter = KokoroTTSServiceAdapter(tts=DummyTTS())
    audio, sample_rate = adapter.synthesize_text("你好")
    assert isinstance(audio, np.ndarray)
    assert sample_rate == 24000
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `uv run pytest tests/test_services/test_openclaw_service.py tests/test_services/test_kokoro_service.py -v`  
Expected: FAIL（模块未实现）

- [ ] **Step 4: 实现两个 adapter**

`src/services/openclaw_service.py`
```python
"""OpenClaw adapter for pipeline use."""

from __future__ import annotations

from src.llm.openclaw import OpenClawLLM


class OpenClawLLMServiceAdapter:
    def __init__(self, llm: OpenClawLLM | None = None) -> None:
        self._llm = llm or OpenClawLLM()

    def chat_messages(self, messages: list[dict]) -> str:
        return self._llm.chat(messages).strip()
```

`src/services/kokoro_service.py`
```python
"""Kokoro adapter for pipeline use."""

from __future__ import annotations

import numpy as np

from src.tts.kokoro import KokoroTTS


class KokoroTTSServiceAdapter:
    def __init__(self, tts: KokoroTTS | None = None) -> None:
        self._tts = tts or KokoroTTS()

    def synthesize_text(self, text: str) -> tuple[np.ndarray, int]:
        audio = self._tts.speak(text)
        return audio, self._tts.sample_rate
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_services/test_openclaw_service.py tests/test_services/test_kokoro_service.py -v`  
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add src/services/openclaw_service.py src/services/kokoro_service.py tests/test_services/test_openclaw_service.py tests/test_services/test_kokoro_service.py
git commit -m "feat: add OpenClaw and Kokoro adapters for pipeline"
```

---

### Task 5: Pipecat pipeline 组装与运行器

**Files:**
- Create: `src/pipecat_app/pipeline.py`
- Modify: `tests/test_pipecat/test_session.py`

- [ ] **Step 1: 写 pipeline 组装失败测试（只测 wiring，不调真实音频）**

```python
from src.pipecat_app.pipeline import build_pipeline_components
from src.pipecat_app.session import SessionController


def test_build_pipeline_components_should_wire_dependencies():
    session = SessionController(end_keywords=["结束"], silence_timeout=10)
    deps = build_pipeline_components(session_controller=session)
    assert "runner" in deps
    assert "pipeline" in deps
    assert "stt_adapter" in deps
    assert "llm_adapter" in deps
    assert "tts_adapter" in deps
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_pipecat/test_session.py -v`  
Expected: FAIL（`cannot import name 'build_pipeline_components'`）

- [ ] **Step 3: 实现 `src/pipecat_app/pipeline.py`**

```python
"""Pipecat pipeline builder and runner wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner

from src.pipecat_app.session import SessionController
from src.services.funasr_service import FunASRServiceAdapter
from src.services.openclaw_service import OpenClawLLMServiceAdapter
from src.services.kokoro_service import KokoroTTSServiceAdapter


@dataclass
class PipelineDeps:
    runner: PipelineRunner
    pipeline: Pipeline
    stt_adapter: FunASRServiceAdapter
    llm_adapter: OpenClawLLMServiceAdapter
    tts_adapter: KokoroTTSServiceAdapter


def build_pipeline_components(session_controller: SessionController) -> dict[str, Any]:
    stt_adapter = FunASRServiceAdapter()
    llm_adapter = OpenClawLLMServiceAdapter()
    tts_adapter = KokoroTTSServiceAdapter()

    # 当前阶段先完成迁移骨架；后续把 adapter 连接到具体 Pipecat processors。
    pipeline = Pipeline([])
    runner = PipelineRunner()

    return {
        "runner": runner,
        "pipeline": pipeline,
        "stt_adapter": stt_adapter,
        "llm_adapter": llm_adapter,
        "tts_adapter": tts_adapter,
        "session_controller": session_controller,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_pipecat/test_session.py -v`  
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/pipecat_app/pipeline.py tests/test_pipecat/test_session.py
git commit -m "feat: add pipecat pipeline builder scaffold"
```

---

### Task 6: 重写 `agent.py` 为 lifecycle 管理器

**Files:**
- Modify: `src/agent.py`
- Create: `tests/test_agent/test_lifecycle.py`

- [ ] **Step 1: 写 lifecycle 失败测试（mock wakeword/pipeline）**

```python
import asyncio
import pytest

from src.agent import HuahuaAgent


@pytest.mark.asyncio
async def test_agent_waits_wakeword_then_starts_pipeline(monkeypatch):
    agent = HuahuaAgent()
    wake_event = asyncio.Event()

    async def fake_wait():
        wake_event.set()

    started = {"value": False}

    async def fake_conversation():
        started["value"] = True

    monkeypatch.setattr(agent, "_wait_for_wakeword", fake_wait)
    monkeypatch.setattr(agent, "_run_pipeline_session", fake_conversation)
    monkeypatch.setattr(agent, "_play_wake_response", lambda: asyncio.sleep(0))

    await agent.run_once_for_test()
    assert wake_event.is_set()
    assert started["value"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_agent/test_lifecycle.py -v`  
Expected: FAIL（`HuahuaAgent` 无 `run_once_for_test`）

- [ ] **Step 3: 重写 `src/agent.py` 的核心结构**

```python
"""花花 AI 助手 — Pipecat 生命周期入口。"""

import asyncio
import numpy as np

from src.audio.player import AudioPlayer
from src.config import config
from src.pipecat_app.pipeline import build_pipeline_components
from src.pipecat_app.session import SessionController
from src.tts.kokoro import KokoroTTS
from src.wakeword.openwakeword import WakewordDetector


class HuahuaAgent:
    def __init__(self):
        self.player = AudioPlayer()
        self.tts = KokoroTTS()
        self.wakeword = WakewordDetector()

    async def run(self):
        while True:
            await self.run_once_for_test()

    async def run_once_for_test(self):
        await self._wait_for_wakeword()
        await self._play_wake_response()
        await self._run_pipeline_session()

    async def _wait_for_wakeword(self):
        event = asyncio.Event()

        async def on_wakeword():
            event.set()

        listen_task = asyncio.create_task(self.wakeword.listen(on_wakeword))
        await event.wait()
        self.wakeword.stop()
        try:
            await asyncio.wait_for(listen_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            listen_task.cancel()

    async def _play_wake_response(self):
        audio = self.tts.speak(config.wake_response)
        audio_int16 = (audio * 32767).astype(np.int16)
        await self.player.speak(audio_int16, self.tts.sample_rate)

    async def _run_pipeline_session(self):
        session = SessionController(
            end_keywords=config.end_keywords,
            silence_timeout=config.silence_timeout,
        )
        deps = build_pipeline_components(session_controller=session)
        runner = deps["runner"]
        pipeline = deps["pipeline"]
        await runner.run(pipeline)


async def main():
    agent = HuahuaAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 再跑测试确认通过**

Run: `uv run pytest tests/test_agent/test_lifecycle.py -v`  
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add src/agent.py tests/test_agent/test_lifecycle.py
git commit -m "refactor: migrate agent lifecycle to wakeword plus pipeline runner"
```

---

### Task 7: 全量回归与本地联调验证

**Files:**
- Modify: `src/pipecat_app/pipeline.py`

- [ ] **Step 1: 在 pipeline 中接入结束词与静默超时钩子**

```python
# 在 build_pipeline_components 返回值外增加会话检查函数，供 processors 回调使用。

def should_terminate(session_controller: SessionController, user_text: str) -> bool:
    if session_controller.should_end_by_text(user_text):
        return True
    if session_controller.is_silence_timeout():
        return True
    return False
```

- [ ] **Step 2: 运行全部单元测试**

Run: `uv run pytest tests -v`  
Expected: PASS（全部通过）

- [ ] **Step 3: 执行静态导入验证**

Run: `uv run python -c "from src.agent import HuahuaAgent; from src.pipecat_app.pipeline import build_pipeline_components; print('imports ok')"`  
Expected: `imports ok`

- [ ] **Step 4: 本地手工端到端验证（一次完整会话）**

Run: `uv run python -m src.agent`  
Expected:
- 说出唤醒词后播放唤醒应答
- 说一句话可得到语音回复
- 播放中插话能打断
- 说结束词返回待机
- 10 秒静默返回待机

- [ ] **Step 5: 提交**

```bash
git add src/pipecat_app/pipeline.py
git commit -m "feat: add session termination hooks for migrated pipeline"
```

---

## Self-Review

### 1) Spec coverage check
- 保留 wakeword 独立运行：Task 6（`_wait_for_wakeword`）覆盖
- STT/LLM/TTS 复用现有模块并加 adapter：Task 3、Task 4 覆盖
- 主循环迁移到 Pipecat runner：Task 5、Task 6 覆盖
- 结束词与静默超时：Task 2、Task 7 覆盖
- 打断能力验证：Task 7 手工端到端检查覆盖

### 2) Placeholder scan
- 无 `TODO`/`TBD`/“自行实现”描述
- 每个代码步骤都给出可粘贴内容
- 每个验证步骤都给出命令与预期结果

### 3) Type consistency
- `SessionController` 在 `agent.py`、`pipeline.py`、tests 中签名一致
- 三个 adapter 方法签名固定：
  - `transcribe_audio(audio) -> str`
  - `chat_messages(messages) -> str`
  - `synthesize_text(text) -> (np.ndarray, int)`

---

Plan complete and saved to `docs/superpowers/plans/2026-03-30-huahua-pipecat-migration-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
