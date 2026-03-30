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

    # FunASR
    funasr_model: str = "paraformer-zh"
    funasr_device: str = "cpu"

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
        self.funasr_model = _get_env("FUNASR_MODEL", self.funasr_model)
        self.funasr_device = _get_env("FUNASR_DEVICE", self.funasr_device)
        self.vad_threshold = float(_get_env("VAD_THRESHOLD", str(self.vad_threshold)))
        self.vad_min_speech_ms = int(_get_env("VAD_MIN_SPEECH_MS", str(self.vad_min_speech_ms)))
        self.wake_response = _get_env("WAKE_RESPONSE", self.wake_response)
        self.sample_rate = int(_get_env("SAMPLE_RATE", str(self.sample_rate)))
        self.chunk_size = int(_get_env("CHUNK_SIZE", str(self.chunk_size)))


config = Config()
