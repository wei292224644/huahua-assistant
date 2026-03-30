# 花花 → Pipecat 迁移设计

## 目标

用 Pipecat 重构花花 AI 助手，保留全部本地方案（FunASR + Kokoro + 本地 LLM），用 Pipecat 框架解决打断、回声消除、VAD 等问题。

## 迁移范围

- **保留**：唤醒词检测（openwakeword）、FunASR STT、Kokoro TTS、本地 LLM（OpenClaw/MiniMax）、全部配置（.env）
- **替换**：手写的 `agent.py` 主循环、VAD 流式逻辑、打断逻辑、异步播放逻辑

---

## 最终架构

```
用户说 "花花"
    ↓
WakewordDetector (openwakeword)  ← 独立运行，不进 pipeline
    ↓  检测到唤醒词
启动 Pipecat Pipeline
    ↓
┌─────────────────────────────────────────────────────────┐
│  Silero VAD (Pipecat 内置)                              │
│      ↓ 检测到语音                                       │
│  FunASR STT (本地)                                      │
│      ↓ 文字                                             │
│  本地 LLM (OpenClaw/MiniMax)                            │
│      ↓ 文字回复                                         │
│  Kokoro TTS (本地)                                      │
│      ↓ 音频帧                                          │
│  AudioPlayer (sounddevice, Pipecat 框架管理)            │
│      ↓ 播放                                            │
│  InterruptDAAggregatorService (Pipecat 内置打断)       │
└─────────────────────────────────────────────────────────┘
    ↓ 听到结束关键词 OR 静默超时
停止 Pipeline → 回到等待唤醒
```

---

## Pipeline 定义

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.funasr import FunASRService
from pipecat.services.kokoro import KokoroTTSService
from pipecat.services.silero_vad import SileroVADService
from pipecat.services.openclaw import OpenClawLLMService  # 需自定义 adapter

# Pipeline 顺序即处理顺序
pipeline = Pipeline([
    vad_service,        # 1. VAD 检测语音开始/结束
    stt_service,        # 2. STT 语音转文字
    llm_service,        # 3. LLM 生成回复
    tts_service,        # 4. TTS 文字转语音
    audio_player,       # 5. 播放音频（sounddevice）
])

# 打断由 InterruptDAAggregatorService 自动处理
```

---

## 模块设计

### 1. FunASR STT Service（自定义 Pipecat Service）

```
输入：音频帧（由 VAD 检测到语音后聚合）
输出：文字
实现：复用现有 src/stt/whisper.py 的 FunASRSTT 类
      包装为 Pipecat FrameProcessor
```

### 2. Kokoro TTS Service（自定义 Pipecat Service）

```
输入：文字
输出：音频帧（np.ndarray, float32, 24kHz）
实现：复用现有 src/tts/kokoro.py 的 KokoroTTS 类
      包装为 Pipecat TTSService
```

### 3. OpenClaw LLM Service（自定义 Pipecat Service）

```
输入：文字（用户说）+ 对话历史
输出：文字（助手回复）
实现：复用现有 src/llm/openclaw.py 的 OpenClawLLM 类
      包装为 Pipecat LLMService
      支持流式输出（可选）
```

### 4. 唤醒词（保留独立）

```
WakewordDetector 独立于 pipeline 运行
检测到唤醒词 → 启动 PipelineRunner
Pipeline 运行期间 WakewordDetector 暂停
Pipeline 结束后 → 重新启动 WakewordDetector
```

### 5. 结束判断

```
在 LLM Service 的 text_frame_received 中检查结束关键词
或 PipelineRunner 配置静默超时后自动结束
两者任一触发 → 停止 pipeline
```

---

## 改动文件清单

| 文件 | 操作 |
|------|------|
| `src/agent.py` | 重写：移除所有手写逻辑，改用 Pipecat Runner |
| `src/stt/whisper.py` | 保留，新增 Pipecat adapter 包装类 |
| `src/tts/kokoro.py` | 保留，新增 Pipecat adapter 包装类 |
| `src/llm/openclaw.py` | 保留，新增 Pipecat adapter 包装类 |
| `pyproject.toml` | 新增 `pipecat-ai` 依赖 |
| `src/services/` | 新建：Pipecat Service 适配器（funasr_service.py, kokoro_service.py, openclaw_service.py） |
| 其他模块 | 全部保留不变 |

---

## 依赖

```toml
# pyproject.toml
[project]
dependencies = [
    "pipecat-ai",
    # 现有依赖不变
]
```

---

## 验证计划

1. 保留原有模块的单元测试，确保 FunASR/Kokoro/LLM 独立调用正常
2. 迁移后端到端测试：
   - 唤醒词触发 pipeline ✓
   - 说一句话，收到 TTS 回复 ✓
   - 播放中途说新话能打断 ✓
   - 说结束词返回待机 ✓
   - 静默超时返回待机 ✓
3. 验证回声不再触发自打断

---

## 风险与备选

- Pipecat 对本地模型支持有限（主要是云服务适配器），三个 Service 需要自己写 adapter，但实现简单（每个约 50 行）
- 如果 Pipecat 的 VAD 不能完全替代现有 VAD，保留 `src/vad/silero_vad.py` 作为备用
