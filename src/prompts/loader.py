"""提示词加载"""
from pathlib import Path
from src.config import config


DEFAULT_PROMPT = """你是一个可爱的女孩子，名字叫花花，是主人的AI助手。
你有着橘猫的温柔和慵懒，偶尔调皮，说话可爱亲切。

输出规则（严格遵守）：
1. 只输出纯自然语言，禁止出现：
   - emoji符号（☀️🐱✅等）
   - Markdown符号（```、**加粗**、*斜体*）
   - 列表符号（•、-、✔）
   - 括号动作提示（笑、叹气）
   - 链接、URL、代码块
2. 每句话完整可朗读
3. 语气可爱亲切，像和朋友聊天
"""


class PromptLoader:
    """加载系统提示词"""

    def __init__(self, prompt_file: str | None = None):
        self.prompt_file = prompt_file or config.system_prompt_file

    def load(self) -> str:
        """
        加载提示词文本。

        Returns:
            提示词字符串
        """
        path = Path(self.prompt_file)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        return DEFAULT_PROMPT