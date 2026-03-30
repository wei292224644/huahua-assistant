"""对话历史管理"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConversationHistory:
    """
    管理对话历史上下文。

    每次用户和助手的对话都记录下来，
    用于后续 LLM 调用时携带完整上下文。
    """

    messages: list[dict] = field(default_factory=list)
    max_history: int = 20  # 最多保留多少轮对话

    def add_user_message(self, text: str):
        """添加用户消息"""
        self.messages.append({
            "role": "user",
            "content": text,
        })
        self._trim()

    def add_assistant_message(self, text: str):
        """添加助手消息"""
        self.messages.append({
            "role": "assistant",
            "content": text,
        })
        self._trim()

    def get_messages_with_system(self, system_prompt: str) -> list[dict]:
        """
        获取包含系统提示词的所有消息。

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        """
        result = [{"role": "system", "content": system_prompt}]
        result.extend(self.messages)
        return result

    def clear(self):
        """清空对话历史"""
        self.messages.clear()

    def _trim(self):
        """修剪过长的历史"""
        # 保留 system + 最近 max_history 条消息
        # system 固定在最前面
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def __len__(self) -> int:
        return len(self.messages)