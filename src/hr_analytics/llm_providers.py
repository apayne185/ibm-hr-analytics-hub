"""Provider-agnostic LLM interface for the "Ask the Data" chat tab.

Every other chatbot module (chat_agent.py, dashboard/app.py) codes against the
LLMProvider protocol and the vendor-neutral Message/ToolCall/ToolResult/ToolSpec
dataclasses defined here -- never against a specific vendor SDK's request or
response shape. Concrete providers (AnthropicProvider, OpenAIProvider) translate
to/from their own wire format internally and lazy-import their SDK inside
__init__, not at module top level, so this module -- and everything that imports
it -- stays importable in environments (like CI) that never install either SDK.
See DECISIONS.md for why this abstraction exists instead of coding directly
against one vendor's API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's arguments


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProviderResponse:
    message: Message
    stop_reason: StopReason
    raw: Any = None  # vendor payload, for the debug panel only -- nothing outside the provider reads it


class LLMProvider(Protocol):
    name: str

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse: ...


class AnthropicProvider:
    """Wraps the Anthropic Messages API. Requires the `anthropic` package
    (install via `uv sync --extra llm`) and ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        import anthropic  # lazy: keeps this module importable without the SDK installed

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        anthropic_messages = [_message_to_anthropic(m) for m in messages]
        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
        ]
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=anthropic_messages,
            tools=anthropic_tools,
            max_tokens=1024,
        )
        return _anthropic_response_to_provider_response(response)


class OpenAIProvider:
    """Wraps the OpenAI Chat Completions API. Requires the `openai` package
    (install via `uv sync --extra llm`) and OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        import openai  # lazy: keeps this module importable without the SDK installed

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        openai_messages = [{"role": "system", "content": system}]
        for m in messages:
            openai_messages.extend(_message_to_openai(m))
        openai_tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            tools=openai_tools or None,
        )
        return _openai_response_to_provider_response(response)


class FakeProvider:
    """Returns a pre-scripted sequence of ProviderResponses, one per call to
    complete() -- makes a full multi-turn tool-calling loop deterministically
    testable with zero network access. Script a [tool_use response, end_turn
    response] pair to simulate the model calling a tool and then answering."""

    name = "fake"

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []  # records what each complete() call was given, for assertions

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if not self._responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        return self._responses.pop(0)


def get_provider() -> LLMProvider | None:
    """Construct a provider from environment configuration, or None if none is
    configured / construction fails for any reason. Never raises -- the
    dashboard's graceful-degradation contract depends on that."""
    choice = os.environ.get("HR_CHAT_PROVIDER")
    if choice is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        else:
            return None
    try:
        if choice == "anthropic":
            return AnthropicProvider()
        if choice == "openai":
            return OpenAIProvider()
    except Exception:
        return None
    return None


def _message_to_anthropic(m: Message) -> dict[str, Any]:
    if m.role == "assistant" and m.tool_calls:
        content: list[dict[str, Any]] = []
        if m.content:
            content.append({"type": "text", "text": m.content})
        content.extend(
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in m.tool_calls
        )
        return {"role": "assistant", "content": content}
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                }
                for tr in m.tool_results
            ],
        }
    return {"role": m.role, "content": m.content}


def _anthropic_response_to_provider_response(response: Any) -> ProviderResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    stop_reason: StopReason = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
    if response.stop_reason == "max_tokens":
        stop_reason = "max_tokens"
    message = Message(role="assistant", content="".join(text_parts), tool_calls=tuple(tool_calls))
    return ProviderResponse(message=message, stop_reason=stop_reason, raw=response)


def _message_to_openai(m: Message) -> list[dict[str, Any]]:
    if m.role == "assistant" and m.tool_calls:
        return [
            {
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": _json_dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ],
            }
        ]
    if m.role == "tool":
        return [{"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content} for tr in m.tool_results]
    return [{"role": m.role, "content": m.content}]


def _openai_response_to_provider_response(response: Any) -> ProviderResponse:
    choice = response.choices[0]
    msg = choice.message
    tool_calls = tuple(
        ToolCall(id=tc.id, name=tc.function.name, arguments=_json_loads(tc.function.arguments))
        for tc in (msg.tool_calls or [])
    )
    stop_reason: StopReason = "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"
    if choice.finish_reason == "length":
        stop_reason = "max_tokens"
    message = Message(role="assistant", content=msg.content or "", tool_calls=tool_calls)
    return ProviderResponse(message=message, stop_reason=stop_reason, raw=response)


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj)


def _json_loads(s: str) -> dict[str, Any]:
    import json

    return json.loads(s)
