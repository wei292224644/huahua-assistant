from src.services.openclaw_service import OpenClawLLMServiceAdapter


class DummyLLM:
    def chat(self, messages):
        assert messages[0]["role"] == "system"
        return "这是回答"


def test_openclaw_adapter_chat():
    adapter = OpenClawLLMServiceAdapter(llm=DummyLLM())
    reply = adapter.chat_messages([{"role": "system", "content": "你是花花"}])
    assert reply == "这是回答"
