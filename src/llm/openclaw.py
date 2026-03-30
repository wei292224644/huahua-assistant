"""OpenClaw LLM 调用"""
from typing import Literal
from openai import OpenAI
from src.config import config


class OpenClawLLM:
    """
    OpenClaw Gateway LLM 调用。

    通过 OpenAI-compatible API 调用 MiniMax。
    """

    def __init__(
        self,
        gateway: str | None = None,
        token: str | None = None,
        model: str | None = None,
    ):
        self.gateway = gateway or config.openclaw_gateway
        self.token = token or config.openclaw_token
        self.model = model or config.llm_model
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.gateway,
                api_key=self.token,
            )
        return self._client

    def chat(self, messages: list[dict]) -> str:
        """
        发送对话消息给 LLM。

        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

        Returns:
            LLM 的回复文字
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_tokens=256,
        )
        return response.choices[0].message.content
