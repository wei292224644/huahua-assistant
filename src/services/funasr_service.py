"""FunASR adapter for pipeline use."""

from __future__ import annotations

import numpy as np

from src.stt.whisper import FunASRSTT


class FunASRServiceAdapter:
    def __init__(self, stt: FunASRSTT | None = None) -> None:
        self._stt = stt or FunASRSTT()

    def transcribe_audio(self, audio: np.ndarray) -> str:
        return self._stt.transcribe(audio).strip()
