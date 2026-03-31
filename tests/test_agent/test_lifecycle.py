import asyncio

import pytest

from src.agent import HuahuaAgent


@pytest.mark.asyncio
async def test_agent_waits_wakeword_then_starts_pipeline(monkeypatch):
    agent = HuahuaAgent()
    wake_event = asyncio.Event()
    started = {"value": False}

    async def fake_wait():
        wake_event.set()

    async def fake_conversation():
        started["value"] = True

    monkeypatch.setattr(agent, "_wait_for_wakeword", fake_wait)
    monkeypatch.setattr(agent, "_run_pipeline_session", fake_conversation)
    monkeypatch.setattr(agent, "_play_wake_response", lambda: asyncio.sleep(0))

    await agent.run_once_for_test()

    assert wake_event.is_set()
    assert started["value"]
