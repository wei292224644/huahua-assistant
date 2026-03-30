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
