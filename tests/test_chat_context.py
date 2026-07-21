from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hr_analytics.chat_context import (
    MAX_CHUNK_CHARS_IN_PROMPT,
    MAX_TOOL_RESULT_CHARS,
    assemble_prompt,
    bound_history,
    format_retrieved_chunks,
    truncate_tool_result,
)
from hr_analytics.llm_providers import Message
from hr_analytics.rag import Chunk


def _messages(n: int) -> list[Message]:
    return [Message(role="user", content=f"message {i}") for i in range(n)]


def test_bound_history_keeps_only_most_recent() -> None:
    history = _messages(20)
    bounded = bound_history(history, max_turns=5)
    assert len(bounded) == 5
    assert bounded[0].content == "message 15"
    assert bounded[-1].content == "message 19"


def test_bound_history_shorter_than_max_returns_all() -> None:
    history = _messages(3)
    assert bound_history(history, max_turns=8) == history


def test_bound_history_zero_max_returns_empty() -> None:
    assert bound_history(_messages(5), max_turns=0) == []


def test_truncate_tool_result_under_limit_unchanged() -> None:
    short = "a short result"
    assert truncate_tool_result(short, max_chars=2000) == short


def test_truncate_tool_result_over_limit_gets_truncated_with_marker() -> None:
    long_result = "x" * (MAX_TOOL_RESULT_CHARS + 500)
    truncated = truncate_tool_result(long_result)
    assert len(truncated) < len(long_result)
    assert "truncated" in truncated
    assert "500" in truncated


def test_format_retrieved_chunks_empty_list() -> None:
    assert format_retrieved_chunks([]) == ""


def test_format_retrieved_chunks_includes_citation_and_score() -> None:
    chunk = Chunk(doc_id="sql_findings.md", section="Overtime", text="Overtime doubles attrition risk.", index=0)
    formatted = format_retrieved_chunks([(chunk, 0.87)])
    assert "sql_findings.md" in formatted
    assert "Overtime" in formatted
    assert "0.87" in formatted
    assert "Overtime doubles attrition risk." in formatted


def test_format_retrieved_chunks_truncates_long_chunk_text() -> None:
    chunk = Chunk(doc_id="d.md", section="s", text="y" * (MAX_CHUNK_CHARS_IN_PROMPT + 200), index=0)
    formatted = format_retrieved_chunks([(chunk, 0.5)])
    assert len(formatted) < MAX_CHUNK_CHARS_IN_PROMPT + 200


def test_assemble_prompt_bounds_history_and_chunks() -> None:
    history = _messages(20)
    chunks = [(Chunk(doc_id=f"d{i}.md", section="s", text="t", index=i), 0.9 - i * 0.1) for i in range(10)]

    snapshot = assemble_prompt("system prompt", history, chunks, max_history_turns=3, max_chunks=2)

    assert len(snapshot.messages) == 3
    assert len(snapshot.retrieved_chunks) == 2
    assert "system prompt" in snapshot.system
    assert snapshot.approx_input_tokens > 0


def test_assemble_prompt_with_no_chunks_omits_context_section() -> None:
    snapshot = assemble_prompt("system prompt", [], [], max_history_turns=8, max_chunks=4)
    assert snapshot.system == "system prompt"
    assert snapshot.retrieved_chunks == []
