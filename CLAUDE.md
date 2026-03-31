# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

花花 (Huahua) is a local voice AI assistant built with Python asyncio. It uses wakeword detection, VAD, STT, LLM, and TTS in an async pipeline.

**Tech Stack:** Python 3.10+, asyncio, sounddevice, funasr (STT), kokoro-onnx (TTS), silero-vad (VAD), openwakeword (wakeword), OpenAI-compatible client (LLM via OpenClaw Gateway → MiniMax)

## Commands

```bash
# Install dependencies
uv sync

# Run the assistant
uv run python -m src.agent

# Run from specific entry
uv run python src/agent.py

# Configure environment
cp .env.example .env
# Then edit .env with your settings (OPENCLAW_TOKEN, etc.)
```

## Architecture

### Pipeline Flow
```
[Microphone] → [Wakeword Detection (openwakeword)] → [Wake Response (Kokoro TTS)]
    ↓
[VAD (Silero VAD)] → [STT (FunASR paraformer-zh)] → [LLM (OpenClaw/MiniMax)] → [TTS (Kokoro)] → [Playback]
    ↑
[Interrupt: VAD detects speech during TTS → stop playback]
```

### State Machine
- **IDLE**: Waiting for wakeword "花花"
- **AWAKE**: Playing wake response
- **CONVERSING**: Full duplex — VAD monitors, user speaks, STT→LLM→TTS loop until end keyword or silence timeout

### Key Files
- `src/agent.py` — Main entry, assembles all modules, runs the async pipeline
- `src/config.py` — Loads `.env` configuration into a `Config` dataclass
- `src/audio/capture.py` — `MicrophoneStream`: async microphone input via sounddevice
- `src/audio/player.py` — `AudioPlayer`: TTS playback with interrupt support
- `src/vad/silero_vad.py` — `SileroVad`: voice activity detection
- `src/stt/whisper.py` — `FunASRSTT`: speech-to-text (FunASR, not faster-whisper)
- `src/tts/kokoro.py` — `KokoroTTS`: text-to-speech (Kokoro-ONNX, 24kHz)
- `src/llm/openclaw.py` — `OpenClawLLM`: OpenAI-compatible LLM calls via OpenClaw gateway
- `src/wakeword/openwakeword.py` — `WakewordDetector`: wakeword detection
- `src/conversation/state_machine.py` — `ConversationStateMachine`: manages IDLE/AWAKE/CONVERSING transitions
- `src/conversation/history.py` — `ConversationHistory`: maintains chat context
- `src/prompts/loader.py` — `PromptLoader`: loads system prompt from file
- `prompts/default.txt` — Default persona prompt (no emoji, markdown, or special formatting in LLM output)

### Module Interface Pattern
Each module is independently importable:
```python
MicrophoneStream()      # async iterable of np.ndarray audio frames
SileroVad.is_speech()   # bool, frame-level speech detection
FunASRSTT.transcribe()  # np.ndarray → str
KokoroTTS.speak()       # str → np.ndarray (float32, -1 to 1), 24kHz
OpenClawLLM.chat()      # list[dict] → str
AudioPlayer.speak()     # async, blocking; speak_async() for interruptible
WakewordDetector.listen() # async, calls callback on detection
```

## Configuration

All config via `.env`. Key variables:
- `HOTWORD` — wake word (default: "花花")
- `END_KEYWORDS` — comma-separated exit keywords
- `SILENCE_TIMEOUT` — seconds of silence before returning to idle
- `OPENCLAW_GATEWAY` — LLM gateway URL (default: `http://localhost:18789/v1`)
- `OPENCLAW_TOKEN` — LLM API token
- `LLM_MODEL` — model name (default: `minimax-portal/MiniMax-M2.7`)
- `KOKORO_VOICE` — TTS voice (default: `zh_female_shaber`)
- `FUNASR_MODEL` — STT model (default: `paraformer-zh`)
- `VAD_THRESHOLD` — VAD sensitivity (default: 0.5)
- `WAKE_RESPONSE` — phrase spoken on wake (default: "诶～我在呀")
- `SAMPLE_RATE` — audio sample rate (default: 16000)
- `CHUNK_SIZE` — microphone chunk size (default: 5120)

## Design Notes

- LLM output must be plain text only (no emoji, markdown, or special formatting) — enforced by system prompt
- Interruption: VAD continuously monitors during TTS playback; if speech is detected, `AudioPlayer.stop()` is called
- Conversation history is cleared on return to IDLE state
- Model files downloaded automatically on first use (no manual download needed)
