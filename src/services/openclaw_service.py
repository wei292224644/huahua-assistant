"""OpenClaw adapter for pipeline use."""

from __future__ import annotations

from src.llm.openclaw import OpenClawLLM


class OpenClawLLMServiceAdapter:
    def __init__(self, llm: OpenClawLLM | None = None) -> None:
        self._llm = llm or OpenClawLLM()

    def chat_messages(self, messages: list[dict]) -> str:
        return self._llm.chat(messages).strip()
