# 花花 AI 助手 — 设计文档

## 1. 概述

花花是一个本地部署的语音 AI 助手。用户通过麦克风与其对话。

**开发阶段（Mac mini M4）：**
- Mac：OpenClaw + agent.py + Whisper/Kokoro（CPU）

**部署阶段（Windows + GPU）：**
- Windows：agent.py + Whisper/Kokoro（GPU）
- Mac：OpenClaw（远程，局域网访问）

STT/VAD/TTS 全部本地运行，只有 LLM 通过 OpenClaw 调用 MiniMax。

---

## 2. 架构

### 2.1 技术选型

| 组件 | 方案 | 说明 |
|------|------|------|
| VAD（语音活动检测） | Silero VAD | 开源免费，CPU 可跑，延迟低 |
| STT（语音识别） | faster-whisper | 本地运行，Mac M4 优化版 |
| LLM | OpenClaw Gateway → MiniMax | 联网调用，已有 token |
| TTS（语音合成） | Kokoro-ONNX | 本地运行，82M，免费 |
| 唤醒词检测 | openWakeWord | 开源免费，中文支持好 |

### 2.2 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      开发阶段（Mac mini M4）                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Mac                                                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ OpenClaw │  │ Whisper  │  │  Kokoro   │            │   │
│  │  │  :18789   │  │  (CPU)   │  │  (CPU)    │            │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │   │
│  │       │              │              │                   │   │
│  │       └──────────────┴──────────────┘                   │   │
│  │                      │                                  │   │
│  │              ┌───────┴───────┐                         │   │
│  │              │   agent.py    │                         │   │
│  │              │   (麦克风)     │                         │   │
│  │              └───────────────┘                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    部署阶段（Windows 2070）                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Windows (GPU)                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Whisper  │  │  Kokoro  │  │   VAD     │            │   │
│  │  │  (GPU)   │  │  (GPU)   │  │  (CPU)    │            │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │   │
│  │       │              │              │                   │   │
│  │       └──────────────┴──────────────┘                   │   │
│  │                      │                                  │   │
│  │              ┌───────┴───────┐                         │   │
│  │              │   agent.py    │                         │   │
│  │              │   (麦克风)     │                         │   │
│  │              └───────────────┘                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                   │
│                    LAN ───┘                                   │
│                           │                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Mac                                                    │   │
│  │  ┌──────────┐                                        │   │
│  │  │ OpenClaw  │  ← http://<mac-ip>:18789/v1          │   │
│  │  │  :18789   │                                        │   │
│  │  └──────────┘                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 数据流

```
[麦克风输入]
    ↓
[Silero VAD] ← 检测是否有人说话
    ↓
[唤醒词检测] ← 检测"花花"
    ↓ 检测到
[Kokoro TTS] → "诶～我在" ← 播放应答
    ↓
[Silero VAD] ← 继续监听
    ↓ 有人说话
[faster-whisper] ← 语音转文字
    ↓
[LLM] ← OpenClaw → MiniMax 生成回复
    ↓
[Kokoro TTS] ← 文字转语音并播放
    ↓
循环，直到检测到结束关键词或静音超时
```

### 2.4 状态机

```
待机状态：
  └→ openWakeWord 持续监听"花花"
  └→ 检测到唤醒词 → 播放应答 → 对话状态

对话状态：
  └→ message_history = []  # 初始化对话历史
  └→ 启动 VAD 监听用户语音
  └→ 用户说完 → STT → LLM → TTS → 播放
  └→ 用户消息加入 history[user]
  └→ LLM 回复加入 history[assistant]
  └→ 继续监听下一轮，带上完整 history 发给 LLM

  # 打断处理（VAD 检测到语音段）
  └→ TTS 播放中 → VAD 检测到新语音 → 打断
      └→ 立即停止 TTS 播放
      └→ 将打断视为新输入，继续对话
      └→ 不清空 history，保持上下文连贯

  └→ 10秒静音超时 → 清空 history → 返回待机
  └→ 听到结束关键词 → 清空 history → 返回待机
```

### 2.5 打断实现逻辑

打断发生在 TTS 播放期间：

```
TTS 播放中：
  ├─ VAD 检测到新语音段开始（用户开始说话）
  ├─ 立即停止当前 TTS 播放（sd.stop()）
  ├─ 等待当前语音段结束
  ├─ 识别打断的语音内容
  ├─ 加入 history，继续对话
  └─ 不产生两段 TTS 重叠播放
```

**打断触发条件**：VAD 检测到 >300ms 的语音段，且 TTS 正在播放

**打断不发生的情况**：
- 静音或噪音时不打断
- TTS 播完后才收到的语音正常处理

### 2.6 对话上下文传递

每次 LLM 调用携带完整对话历史：

```
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    ...message_history  # 所有历史消息
]
response = llm.chat(messages)
```

用户说"今天天气怎么样" → history + [user: 今天天气怎么样]
花花回答"晴天" → history + [assistant: 晴天]
用户说"适合出门吗" → history + [user: 今天天气怎么样] + [assistant: 晴天] + [user: 适合出门吗]
→ LLM 知道"今天天气晴天"，回答更准确

---

## 3. 组件设计

### 3.1 唤醒检测

- 使用 `openWakeWord` 库，唤醒词为"花花"
- 麦克风音频流持续输入 openWakeWord
- 检测到唤醒词后触发回调

### 3.2 对话循环

每次用户说话（VAD 检测到语音段结束后）：
1. `faster-whisper` 将语音转为文字
2. 发送给 LLM，获取回复
3. `Kokoro TTS` 将回复转为语音并播放
4. 继续监听下一轮

### 3.3 结束条件

- **静音超时**：用户说完后 10 秒内没有新语音，自动返回待机
- **结束关键词**：`["再见", "退下", "好了", "不用了", "结束", "拜拜"]`

---

## 4. 代码结构

```
huahua-assistant/
├── agent.py              # 主程序入口
├── pyproject.toml        # uv 项目配置
├── .env.example          # 环境变量示例
├── models/               # 模型文件目录（预下载）
│   ├── kokoro/          # Kokoro 音色模型
│   └── whisper/         # Whisper 模型（可选）
├── prompts/              # 提示词文件目录
│   └── default.txt       # 默认花花人设提示词
├── README.md
└── docs/
    └── specs/
        └── 2026-03-29-huahua-ai-assistant-design.md
```

### 4.1 pyproject.toml

```toml
[project]
name = "huahua-assistant"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "faster-whisper",
    "kokoro-onnx",
    "openwakeword",
    "silero-vad",
    "sounddevice",
    "numpy",
    "python-dotenv",
    "openai",
]
```

### 4.1 代码模块化要求

每个模块职责单一，接口清晰，可独立测试：

```
huahua/
├── __init__.py
├── agent.py                 # 主入口，组装各模块
├── config.py                # .env 配置加载
├── audio/
│   ├── __init__.py
│   ├── capture.py           # MicrophoneStream — 麦克风音频捕获
│   └── player.py             # AudioPlayer — TTS 播放 + 打断
├── vad/
│   ├── __init__.py
│   └── silero_vad.py        # SileroVad — 语音活动检测
├── stt/
│   ├── __init__.py
│   └── whisper.py           # WhisperSTT — 语音转文字
├── tts/
│   ├── __init__.py
│   └── kokoro.py            # KokoroTTS — 文字转语音
├── llm/
│   ├── __init__.py
│   └── openclaw.py          # OpenClawLLM — LLM 调用
├── wakeword/
│   ├── __init__.py
│   └── openwakeword.py      # WakewordDetector — 唤醒词检测
├── conversation/
│   ├── __init__.py
│   ├── state_machine.py     # ConversationState — 状态机
│   └── history.py            # ConversationHistory — 对话历史管理
├── prompts/
│   ├── __init__.py
│   └── loader.py             # PromptLoader — 提示词加载
└── utils/
    ├── __init__.py
    └── typing.py             # 类型定义
```

**模块接口示例：**

```python
# audio/capture.py
class MicrophoneStream:
    async def __aiter__(self): ...
    async def get_audio(self) -> np.ndarray: ...

# vad/silero_vad.py
class SileroVad:
    def is_speech(self, audio: np.ndarray) -> bool: ...
    async def detect_speech_segments(self, audio_stream): ...

# stt/whisper.py
class WhisperSTT:
    def transcribe(self, audio: np.ndarray) -> str: ...

# tts/kokoro.py
class KokoroTTS:
    def speak(self, text: str) -> None: ...  # 同步播放
    def speak_async(self, text: str) -> None: ...  # 异步播放，可打断
    def stop(self) -> None: ...

# llm/openclaw.py
class OpenClawLLM:
    def chat(self, messages: list[dict]) -> str: ...

# wakeword/openwakeword.py
class WakewordDetector:
    async def listen(self, callback): ...  # 检测到唤醒词时调用 callback
```

**代码要求：**
- 每个模块可独立 import 和测试
- 模块之间通过明确定义的接口通信，不直接传递内部状态
- 使用 asyncio 构建异步 pipeline
- 打字简洁，避免过度封装

---

## 5. 人设与输出规则

### 5.1 系统提示词

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

### 5.2 音色

| 音色 ID | 风格 |
|--------|------|
| `zh_female_shaber` | 自然女声，中文最佳（推荐） |
| `zh_female_xiaobei` | 温柔女声 |
| `zh_female_bao` | 活泼女声 |

---

## 6. 环境配置

### 6.1 .env 配置文件

```bash
# === 唤醒与结束配置 ===
HOTWORD=花花                    # 唤醒词
END_KEYWORDS=再见,退下,好了,不用了,结束,拜拜  # 结束关键词，逗号分隔
SILENCE_TIMEOUT=10              # 静音超时（秒）

# === OpenClaw LLM ===
OPENCLAW_GATEWAY=http://localhost:18789/v1   # Mac 开发用
# OPENCLAW_GATEWAY=http://192.168.x.x:18789/v1  # Windows 部署用，填 Mac 的局域网 IP
OPENCLAW_TOKEN=your_token_here
LLM_MODEL=minimax-portal/MiniMax-M2.7

# === TTS 配置 ===
KOKORO_VOICE=zh_female_shaber   # Kokoro 音色
KOKORO_MODEL_PATH=              # 可选，自定义音色模型路径

# === 系统提示词（可指定文件路径）===
SYSTEM_PROMPT_FILE=            # 可选，指定提示词文件路径，默认使用内置花花人设
SYSTEM_PROMPT=你是一个可爱的女孩子...  # 可选，直接写提示词，优先于 FILE

# === Whisper 配置 ===
WHISPER_MODEL=base              # faster-whisper 模型大小
WHISPER_DEVICE=cpu              # mac: cpu, windows: cuda

# === VAD 配置 ===
VAD_THRESHOLD=0.5               # Silero VAD 灵敏度阈值
VAD_MIN_SPEECH_MS=250           # 最小语音段时长（毫秒）

# === 唤醒应答 ===
WAKE_RESPONSE=诶～我在呀          # 唤醒后播放的应答语

# === 音频配置 ===
SAMPLE_RATE=16000               # 采样率
CHUNK_SIZE=5120                 # 麦克风每次读取的帧数
```

### 6.2 .env.example

```bash
cp .env.example .env
# 编辑 .env 填入实际值
```

### 6.3 开发阶段（Mac mini M4）

- **faster-whisper**：`uv pip install faster-whisper`，Mac M4 可用
- **Kokoro**：`uv pip install kokoro-onnx`
- **openWakeWord**：`uv pip install openwakeword`
- **Silero VAD**：`uv pip install silero-vad`
- OpenClaw：本地 `localhost:18789`

模型文件通过 pip 安装时自动下载，或提前放入 `models/` 目录。

### 6.4 部署阶段（Windows + GPU）

- **faster-whisper**：`uv pip install faster-whisper[gpu]`，GPU 加速
- **Kokoro**：`uv pip install kokoro-onnx`，GPU 加速
- **openWakeWord**：`uv pip install openwakeword`
- **Silero VAD**：`uv pip install silero-vad`
- **OpenClaw**：运行在 Mac 上，Windows 通过局域网访问 `http://<mac-ip>:18789/v1`

模型文件提前下载到 `models/` 目录，代码里指定本地路径加载。

---

## 7. 实施步骤

### 7.1 开发阶段（Mac mini M4）

```bash
# 1. 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖（uv 自动管理虚拟环境）
uv sync

# 3. 验证安装
python -c "from kokoro import KModel; print('Kokoro OK')"
python -c "import openwakeword; print('openWakeWord OK')"

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env：
#   OPENCLAW_GATEWAY=http://localhost:18789/v1

# 5. 启动 OpenClaw（单独终端）
openclaw

# 6. 运行花花
python agent.py
```

### 7.2 部署阶段（Windows + GPU）

```powershell
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv pip install faster-whisper[gpu] kokoro-onnx openwakeword silero-vad
uv pip install sounddevice numpy python-dotenv openai

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env：
#   OPENCLAW_GATEWAY=http://<mac-ip>:18789/v1
#   WHISPER_DEVICE=cuda
#   KOKORO_MODEL_PATH=models/kokoro/

# 4. 确认 OpenClaw 在 Mac 上运行

# 5. 运行花花
python agent.py
```

---

## 8. 成本

| 项目 | 费用 |
|------|------|
| Silero VAD | 免费（开源） |
| faster-whisper | 免费（开源） |
| Kokoro TTS | 免费（开源 82M） |
| openWakeWord | 免费（开源） |
| LLM | 复用 OpenClaw（MiniMax） |
| **总计** | **零成本** |

---

## 9. 硬件性能

| 机器 | VAD | STT | TTS | LLM |
|------|-----|-----|-----|-----|
| Mac mini M4 | ✅ CPU 流畅 | ✅ M4 优化，很快 | ✅ CPU 流畅 | ✅ OpenClaw 本地 |
| Windows 2070 16G | ✅ CPU 流畅 | ✅ GPU 加速 <500ms | ✅ GPU 更流畅 | ⏱️ 局域网调用 Mac |

---

## 10. 待验证点

- [ ] openWakeWord 对"花花"的识别准确率（可能需要调灵敏度）
- [ ] Silero VAD 在 Mac 上的延迟表现
- [ ] faster-whisper M4 优化版的实际速度
- [ ] Kokoro 中文音色在实际对话中的自然度
