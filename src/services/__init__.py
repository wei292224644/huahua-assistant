"""Adapters that bridge local modules to Pipecat pipeline."""

from src.services.funasr_service import FunASRServiceAdapter
from src.services.kokoro_service import KokoroTTSServiceAdapter
from src.services.openclaw_service import OpenClawLLMServiceAdapter

__all__ = [
    "FunASRServiceAdapter",
    "KokoroTTSServiceAdapter",
    "OpenClawLLMServiceAdapter",
]
