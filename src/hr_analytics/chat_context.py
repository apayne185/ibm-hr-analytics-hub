"""Context-window management for the "Ask the Data" chat agent.

Every bound here is a named module-level constant -- deliberately, so there's a
single, pointable-to answer to "how do you manage the context window" instead of
magic numbers scattered through chat_agent.py. Pruning is simple truncation (drop
the oldest turns, cap chunk/tool-result sizes), not summarization or compaction,
and deliberately NOT any vendor's server-side context-editing feature (e.g.
Anthropic's compaction beta) -- that would couple the strategy to one provider and
defeat the point of llm_providers.py's provider-agnostic abstraction. See
DECISIONS.md for the full reasoning.

assemble_prompt() produces a PromptSnapshot -- the object the dashboard's debug
panel renders directly, so what actually went into the prompt is inspectable,
not a black box.
"""

from __future__ import annotations

from dataclasses import dataclass

from hr_analytics.llm_providers import Message
from hr_analytics.rag import Chunk

MAX_HISTORY_TURNS = 8
MAX_RETRIEVED_CHUNKS = 4
MAX_CHUNK_CHARS_IN_PROMPT = 500
MAX_TOOL_RESULT_CHARS = 2000

# A rough, provider-agnostic estimate for the debug panel only -- NOT exact
# tokenization (tiktoken is the wrong tokenizer for Claude, and this module
# must stay provider-agnostic; a provider could optionally expose an exact
# count_tokens() later, but nothing here depends on it).
APPROX_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class PromptSnapshot:
    system: str
    messages: list[Message]
    retrieved_chunks: list[tuple[Chunk, float]]
    approx_input_tokens: int


def bound_history(history: list[Message], max_turns: int = MAX_HISTORY_TURNS) -> list[Message]:
    """Keep only the most recent max_turns messages. Simple truncation, not
    summarization -- sufficient at this app's conversation lengths."""
    if max_turns <= 0:
        return []
    return history[-max_turns:]


def truncate_tool_result(content: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"... [truncated, {len(content) - max_chars} more characters]"


def format_retrieved_chunks(
    chunks: list[tuple[Chunk, float]], max_chars: int = MAX_CHUNK_CHARS_IN_PROMPT
) -> str:
    """Render retrieved chunks into the block of text inserted into the
    system/context prompt, each truncated and clearly cited by source."""
    if not chunks:
        return ""
    blocks = []
    for chunk, score in chunks:
        text = chunk.text if len(chunk.text) <= max_chars else chunk.text[:max_chars] + "..."
        blocks.append(f"[{chunk.doc_id} § {chunk.section} (relevance {score:.2f})]\n{text}")
    return "\n\n".join(blocks)


def _approx_tokens(*texts: str) -> int:
    return sum(len(t) for t in texts) // APPROX_CHARS_PER_TOKEN


def assemble_prompt(
    system: str,
    history: list[Message],
    retrieved_chunks: list[tuple[Chunk, float]],
    max_history_turns: int = MAX_HISTORY_TURNS,
    max_chunks: int = MAX_RETRIEVED_CHUNKS,
) -> PromptSnapshot:
    bounded_history = bound_history(history, max_history_turns)
    bounded_chunks = retrieved_chunks[:max_chunks]

    chunks_block = format_retrieved_chunks(bounded_chunks)
    full_system = system if not chunks_block else f"{system}\n\nRelevant context:\n{chunks_block}"

    approx_tokens = _approx_tokens(full_system, *(m.content for m in bounded_history))

    return PromptSnapshot(
        system=full_system,
        messages=bounded_history,
        retrieved_chunks=bounded_chunks,
        approx_input_tokens=approx_tokens,
    )
