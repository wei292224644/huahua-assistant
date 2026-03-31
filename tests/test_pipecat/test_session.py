import asyncio

import pytest

from src.pipecat_app.pipeline import build_pipeline_components, should_terminate
from src.pipecat_app.session import SessionController


def test_session_should_end_by_keyword():
    controller = SessionController(end_keywords=["结束", "拜拜"], silence_timeout=10)
    assert controller.should_end_by_text("我们结束吧")
    assert not controller.should_end_by_text("继续聊")


@pytest.mark.asyncio
async def test_session_should_end_by_timeout():
    controller = SessionController(end_keywords=["结束"], silence_timeout=0)
    controller.mark_user_activity()
    await asyncio.sleep(0.01)
    assert controller.is_silence_timeout()


def test_build_pipeline_components_should_wire_dependencies():
    session = SessionController(end_keywords=["结束"], silence_timeout=10)
    deps = build_pipeline_components(session_controller=session)
    assert "runner" in deps
    assert "pipeline" in deps
    assert "stt_adapter" in deps
    assert "llm_adapter" in deps
    assert "tts_adapter" in deps


def test_should_terminate_by_keyword():
    session = SessionController(end_keywords=["结束"], silence_timeout=10)
    assert should_terminate(session, "我们结束吧")
    assert not should_terminate(session, "继续聊")


@pytest.mark.asyncio
async def test_should_terminate_by_silence_timeout():
    session = SessionController(end_keywords=["结束"], silence_timeout=0)
    session.mark_user_activity()
    await asyncio.sleep(0.01)
    assert should_terminate(session, "继续聊")
