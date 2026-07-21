from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import hr_analytics.chat_agent as chat_agent
import hr_analytics.load_db as load_db
import hr_analytics.sql_tool as sql_tool
from hr_analytics.chat_agent import MAX_TOOL_ITERATIONS, run_turn
from hr_analytics.llm_providers import FakeProvider, Message, ProviderResponse, ToolCall
from hr_analytics.rag import build_index_from_chunks, chunk_markdown


@pytest.fixture
def db_path(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh temp db, same pattern as test_sql_queries.py/test_sql_tool.py,
    but also points sql_tool.DB_PATH at it since chat_agent's tools call
    sql_tool functions using their module-level default DB_PATH."""
    tmp_path = tmp_path_factory.mktemp("chat_agent_test_db")
    db = tmp_path / "test.db"

    import hr_analytics.synthetic_hiring as synthetic_hiring

    raw = pd.read_csv(load_db.RAW_PATH, encoding="utf-8-sig")
    pipeline = synthetic_hiring.build_synthetic_hiring_pipeline(raw)
    pipeline_path = tmp_path / "pipeline.csv"
    pipeline.to_csv(pipeline_path, index=False)

    orig = (load_db.PIPELINE_PATH, load_db.SCHEMA_PATH, load_db.DB_PATH)
    load_db.PIPELINE_PATH = pipeline_path
    load_db.SCHEMA_PATH = REPO_ROOT / "sql/schema.sql"
    load_db.DB_PATH = db
    try:
        load_db.main()
    finally:
        load_db.PIPELINE_PATH, load_db.SCHEMA_PATH, load_db.DB_PATH = orig

    monkeypatch.setattr(sql_tool, "DB_PATH", db)
    return db


@pytest.fixture
def empty_index():
    return build_index_from_chunks([])


@pytest.fixture
def sample_index():
    chunks = chunk_markdown(
        "sql_findings.md",
        "## Overtime\n\nOvertime roughly doubles the hazard of attrition.\n\n"
        "## Tenure\n\nAttrition is front-loaded in the first two years.\n",
    )
    return build_index_from_chunks(chunks)


def test_run_turn_immediate_answer_no_tools(db_path, empty_index) -> None:
    provider = FakeProvider(
        [ProviderResponse(message=Message(role="assistant", content="Hello!"), stop_reason="end_turn")]
    )
    reply, snapshot = run_turn(provider, empty_index, [], "hi")
    assert reply.content == "Hello!"
    assert snapshot.retrieved_chunks == []


def test_run_turn_get_attrition_rate_tool_call(db_path, empty_index) -> None:
    tool_use = ProviderResponse(
        message=Message(
            role="assistant",
            tool_calls=(ToolCall(id="c1", name="get_attrition_rate", arguments={"department": "Sales"}),),
        ),
        stop_reason="tool_use",
    )
    final = ProviderResponse(
        message=Message(role="assistant", content="Sales attrition is roughly 40%."), stop_reason="end_turn"
    )
    provider = FakeProvider([tool_use, final])

    reply, snapshot = run_turn(provider, empty_index, [], "what's the attrition rate in sales?")

    assert reply.content == "Sales attrition is roughly 40%."
    # second complete() call must have received the tool result as a message
    second_call_messages = provider.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "attrition_rate_pct" in tool_messages[0].tool_results[0].content


def test_run_turn_search_docs_accumulates_retrieved_chunks(db_path, sample_index) -> None:
    tool_use = ProviderResponse(
        message=Message(
            role="assistant",
            tool_calls=(ToolCall(id="c1", name="search_docs", arguments={"query": "overtime hazard"}),),
        ),
        stop_reason="tool_use",
    )
    final = ProviderResponse(
        message=Message(role="assistant", content="Overtime roughly doubles attrition risk."), stop_reason="end_turn"
    )
    provider = FakeProvider([tool_use, final])

    reply, snapshot = run_turn(provider, sample_index, [], "does overtime matter?")

    assert reply.content == "Overtime roughly doubles attrition risk."
    assert len(snapshot.retrieved_chunks) > 0
    assert snapshot.retrieved_chunks[0][0].section == "Overtime"


def test_run_turn_sql_query_tool_call(db_path, empty_index) -> None:
    tool_use = ProviderResponse(
        message=Message(
            role="assistant",
            tool_calls=(ToolCall(id="c1", name="sql_query", arguments={"sql": "SELECT COUNT(*) AS n FROM employees"}),),
        ),
        stop_reason="tool_use",
    )
    final = ProviderResponse(message=Message(role="assistant", content="There are 1470 employees."), stop_reason="end_turn")
    provider = FakeProvider([tool_use, final])

    reply, snapshot = run_turn(provider, empty_index, [], "how many employees are there?")
    assert reply.content == "There are 1470 employees."


def test_run_turn_unsafe_sql_query_returns_error_not_crash(db_path, empty_index) -> None:
    tool_use = ProviderResponse(
        message=Message(
            role="assistant",
            tool_calls=(ToolCall(id="c1", name="sql_query", arguments={"sql": "DROP TABLE employees"}),),
        ),
        stop_reason="tool_use",
    )
    final = ProviderResponse(message=Message(role="assistant", content="I can't run that query."), stop_reason="end_turn")
    provider = FakeProvider([tool_use, final])

    reply, snapshot = run_turn(provider, empty_index, [], "drop the employees table")

    assert reply.content == "I can't run that query."
    second_call_messages = provider.calls[1]["messages"]
    tool_message = [m for m in second_call_messages if m.role == "tool"][0]
    assert tool_message.tool_results[0].is_error is True


def test_run_turn_unknown_tool_name_returns_error_not_crash(db_path, empty_index) -> None:
    tool_use = ProviderResponse(
        message=Message(role="assistant", tool_calls=(ToolCall(id="c1", name="not_a_real_tool", arguments={}),)),
        stop_reason="tool_use",
    )
    final = ProviderResponse(message=Message(role="assistant", content="Something went wrong."), stop_reason="end_turn")
    provider = FakeProvider([tool_use, final])

    reply, snapshot = run_turn(provider, empty_index, [], "trigger unknown tool")
    assert reply.content == "Something went wrong."


def test_run_turn_flight_risk_watchlist_tool_call(db_path, empty_index, tmp_path, monkeypatch) -> None:
    risk_csv = tmp_path / "predicted_attrition_risk.csv"
    pd.DataFrame(
        [
            {"EmployeeNumber": 1, "Department": "Sales", "JobRole": "Sales Rep", "Attrition": "No", "predicted_hazard_score": 5.0, "risk_percentile": 0.99},
            {"EmployeeNumber": 2, "Department": "Sales", "JobRole": "Sales Rep", "Attrition": "No", "predicted_hazard_score": 1.0, "risk_percentile": 0.2},
            {"EmployeeNumber": 3, "Department": "Sales", "JobRole": "Sales Rep", "Attrition": "Yes", "predicted_hazard_score": 9.0, "risk_percentile": None},
        ]
    ).to_csv(risk_csv, index=False)
    monkeypatch.setattr(chat_agent, "RISK_PATH", risk_csv)

    tool_use = ProviderResponse(
        message=Message(
            role="assistant",
            tool_calls=(ToolCall(id="c1", name="get_flight_risk_watchlist", arguments={"department": "Sales", "top_n": 5}),),
        ),
        stop_reason="tool_use",
    )
    final = ProviderResponse(message=Message(role="assistant", content="Employee 1 is highest risk."), stop_reason="end_turn")
    provider = FakeProvider([tool_use, final])

    reply, snapshot = run_turn(provider, empty_index, [], "who's at flight risk in sales?")

    assert reply.content == "Employee 1 is highest risk."
    tool_message = [m for m in provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "1" in tool_message.tool_results[0].content
    assert "3" not in tool_message.tool_results[0].content  # leaver excluded


def test_run_turn_stops_at_max_tool_iterations(db_path, empty_index) -> None:
    """A model that never stops calling tools must not loop forever."""
    always_tool_use = ProviderResponse(
        message=Message(role="assistant", tool_calls=(ToolCall(id="c1", name="get_attrition_rate", arguments={}),)),
        stop_reason="tool_use",
    )
    provider = FakeProvider([always_tool_use] * MAX_TOOL_ITERATIONS)

    reply, snapshot = run_turn(provider, empty_index, [], "loop forever")

    assert len(provider.calls) == MAX_TOOL_ITERATIONS
    assert "tool-call limit" in reply.content
