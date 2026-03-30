"""对话状态机"""
import asyncio
from enum import Enum
from typing import Callable, Awaitable
from src.conversation.history import ConversationHistory
from src.utils.typing import ConversationState as StateEnum


class ConversationStateMachine:
    """
    对话状态机。

    管理待机、唤醒、对话状态的转换。
    """

    def __init__(self):
        self.state: StateEnum = StateEnum.IDLE
        self.history: ConversationHistory = ConversationHistory()
        self._state_lock = asyncio.Lock()

    async def transition_to(self, new_state: StateEnum):
        """切换状态"""
        async with self._state_lock:
            old_state = self.state
            self.state = new_state

            if new_state == StateEnum.IDLE:
                self.history.clear()

            print(f"状态切换: {old_state.value} -> {new_state.value}")

    async def is_idle(self) -> bool:
        return self.state == StateEnum.IDLE

    async def is_awake(self) -> bool:
        return self.state == StateEnum.AWAKE

    async def is_conversing(self) -> bool:
        return self.state == StateEnum.CONVERSING
