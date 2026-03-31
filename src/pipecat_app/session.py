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

    @property
    def silence_timeout(self) -> int:
        return self._silence_timeout
