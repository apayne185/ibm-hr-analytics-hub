from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hr_analytics.rag import (
    build_index,
    build_index_from_chunks,
    chunk_decisions_log,
    chunk_markdown,
    chunk_plain_text,
    retrieve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


SAMPLE_MARKDOWN = """# Title

Intro text that should be dropped since it's before the first header.

## First Section

This is the first section's content. It talks about overtime and attrition.

## Second Section

This is the second section. It talks about income and job level.
"""


def test_chunk_markdown_splits_on_headers_and_drops_preamble() -> None:
    chunks = chunk_markdown("sample.md", SAMPLE_MARKDOWN)
    assert len(chunks) == 2
    assert chunks[0].section == "First Section"
    assert "overtime" in chunks[0].text
    assert chunks[1].section == "Second Section"
    assert "income" in chunks[1].text
    assert all(c.doc_id == "sample.md" for c in chunks)


def test_chunk_markdown_oversized_section_gets_split_with_overlap() -> None:
    long_body = "word " * 400  # ~2000 chars, well over MAX_CHUNK_CHARS=800
    text = f"## Big Section\n\n{long_body}"
    chunks = chunk_markdown("big.md", text)
    assert len(chunks) > 1
    assert all(c.section == "Big Section" for c in chunks)
    assert all(len(c.text) <= 800 for c in chunks)


def test_chunk_decisions_log_splits_on_dated_entries() -> None:
    text = (
        "# Decisions Log\n\n"
        "## 2026-01-01 — First decision\n\n"
        "**Decision:** did the first thing.\n\n"
        "## 2026-01-02 — Second decision\n\n"
        "**Decision:** did the second thing.\n"
    )
    chunks = chunk_decisions_log(text)
    assert len(chunks) == 2
    assert "2026-01-01" in chunks[0].section
    assert "first thing" in chunks[0].text
    assert "2026-01-02" in chunks[1].section


def test_chunk_plain_text_single_chunk_for_small_file() -> None:
    chunks = chunk_plain_text("notes.txt", "short plain text content")
    assert len(chunks) == 1
    assert chunks[0].section == "(whole file)"


def test_chunk_plain_text_empty_file_produces_no_chunks() -> None:
    assert chunk_plain_text("empty.txt", "   \n  ") == []


def test_retrieval_ranks_on_topic_chunk_first() -> None:
    chunks = chunk_markdown(
        "fixture.md",
        "## Cooking\n\nHow to bake bread and make pasta from scratch.\n\n"
        "## Astronomy\n\nStars, planets, and galaxies in the night sky.\n\n"
        "## Gardening\n\nHow to grow tomatoes and plant a vegetable garden.\n",
    )
    index = build_index_from_chunks(chunks)

    results = retrieve(index, "how do I grow vegetables in my garden", k=1)
    assert len(results) == 1
    assert results[0][0].section == "Gardening"

    results = retrieve(index, "tell me about planets and stars", k=1)
    assert len(results) == 1
    assert results[0][0].section == "Astronomy"


def test_retrieval_respects_min_score_threshold() -> None:
    chunks = chunk_markdown("fixture.md", "## Only Section\n\nSomething about zebras and giraffes.\n")
    index = build_index_from_chunks(chunks)
    results = retrieve(index, "completely unrelated query about aerospace engineering", k=4, min_score=0.99)
    assert results == []


def test_retrieve_on_empty_index_returns_empty() -> None:
    index = build_index_from_chunks([])
    assert retrieve(index, "anything", k=4) == []


def test_build_index_over_real_docs_produces_chunks() -> None:
    """End-to-end sanity check against the actual repo docs (no network,
    just local file reads + TF-IDF fit)."""
    index = build_index(docs_dir=REPO_ROOT / "docs", decisions_path=REPO_ROOT / "DECISIONS.md")
    assert len(index.chunks) > 5
    doc_ids = {c.doc_id for c in index.chunks}
    assert "DECISIONS.md" in doc_ids
    assert "sql_findings.md" in doc_ids
    assert "survival_model_findings.md" in doc_ids

    results = retrieve(index, "overtime attrition hazard ratio", k=3)
    assert len(results) > 0
