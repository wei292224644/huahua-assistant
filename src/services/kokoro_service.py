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
