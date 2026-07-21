from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hr_analytics.llm_providers import (
    FakeProvider,
    Message,
    ProviderResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
    get_provider,
)


def test_fake_provider_returns_scripted_responses_in_order() -> None:
    r1 = ProviderResponse(message=Message(role="assistant", content="first"), stop_reason="end_turn")
    r2 = ProviderResponse(message=Message(role="assistant", content="second"), stop_reason="end_turn")
    provider = FakeProvider([r1, r2])

    assert provider.complete(system="s", messages=[], tools=[]) is r1
    assert provider.complete(system="s", messages=[], tools=[]) is r2


def test_fake_provider_raises_when_exhausted() -> None:
    provider = FakeProvider([])
    try:
        provider.complete(system="s", messages=[], tools=[])
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_fake_provider_records_calls_for_assertions() -> None:
    provider = FakeProvider([ProviderResponse(message=Message(role="assistant", content="ok"), stop_reason="end_turn")])
    tool = ToolSpec(name="search_docs", description="d", parameters={"type": "object", "properties": {}})
    provider.complete(system="sys prompt", messages=[Message(role="user", content="hi")], tools=[tool])

    assert len(provider.calls) == 1
    assert provider.calls[0]["system"] == "sys prompt"
    assert provider.calls[0]["tools"] == [tool]


def test_multi_turn_tool_use_then_end_turn_sequence() -> None:
    """The shape a real tool-calling loop drives: model asks for a tool call,
    then (once fed the result) answers with plain text."""
    tool_use_response = ProviderResponse(
        message=Message(
            role="assistant",
            tool_calls=(ToolCall(id="call_1", name="get_attrition_rate", arguments={"department": "Sales"}),),
        ),
        stop_reason="tool_use",
    )
    final_response = ProviderResponse(
        message=Message(role="assistant", content="Sales attrition is 39.8%."), stop_reason="end_turn"
    )
    provider = FakeProvider([tool_use_response, final_response])

    first = provider.complete(system="s", messages=[Message(role="user", content="attrition in sales?")], tools=[])
    assert first.stop_reason == "tool_use"
    assert first.message.tool_calls[0].name == "get_attrition_rate"

    tool_result_message = Message(
        role="tool", tool_results=(ToolResult(tool_call_id="call_1", content="39.8%"),)
    )
    second = provider.complete(
        system="s", messages=[Message(role="user", content="attrition in sales?"), tool_result_message], tools=[]
    )
    assert second.stop_reason == "end_turn"
    assert "39.8" in second.message.content


def test_get_provider_returns_none_with_no_env_configured(monkeypatch) -> None:
    monkeypatch.delenv("HR_CHAT_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_provider() is None


def test_get_provider_degrades_gracefully_when_sdk_not_installed(monkeypatch) -> None:
    """anthropic/openai are optional extras -- this test environment (like CI)
    never installs them, so get_provider() must return None instead of raising
    ImportError. This only proves something if the SDK is actually absent, so
    guard against a false pass if a future change installs it as a hard dep."""
    import importlib.util

    assert importlib.util.find_spec("anthropic") is None, "test assumes anthropic is not installed"
    monkeypatch.setenv("HR_CHAT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    assert get_provider() is None
