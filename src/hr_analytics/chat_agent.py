"""The tool-calling loop for the "Ask the Data" chat agent.

Ties together llm_providers.py (the model), rag.py (doc retrieval), sql_tool.py
(database access), and chat_context.py (context-window bounding) into one
run_turn() call per user message.

The tool roster is deliberately both parameterized *and* free-form, to
demonstrate that tradeoff was evaluated, not picked dogmatically:
- search_docs: agent-orchestrated retrieval (a tool call), not a mandatory
  prepend before every turn -- the model decides per-turn whether a question
  needs doc context, avoiding wasted context on purely quantitative questions
  like "how many people are in Sales?". This is "agentic RAG" over naive
  always-retrieve.
- get_attrition_rate / get_flight_risk_watchlist: small parameterized tools,
  zero injection surface (sql_tool.execute_parameterized with '?' bindings),
  fastest path for the highest-frequency questions.
- sql_query: the constrained free-form fallback (sql_tool.run_read_only_query)
  for anything the parameterized tools don't cover.

See DECISIONS.md for the full reasoning behind this design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hr_analytics import sql_tool
from hr_analytics.chat_context import PromptSnapshot, assemble_prompt, truncate_tool_result
from hr_analytics.llm_providers import LLMProvider, Message, ToolCall, ToolResult, ToolSpec
from hr_analytics.rag import RetrievalIndex, retrieve

RISK_PATH = Path("data/processed/predicted_attrition_risk.csv")
DEFAULT_WATCHLIST_TOP_N = 10
MAX_TOOL_ITERATIONS = 5  # guards against a runaway tool-calling loop

SYSTEM_PROMPT = (
    "You are an analyst assistant for the IBM HR Analytics Hub, a people-analytics "
    "portfolio project covering employee attrition at a fictional company. Answer "
    "using the tools available: search_docs for methodology/qualitative questions "
    "(SQL findings, survival model results, design decisions), get_attrition_rate "
    "or get_flight_risk_watchlist for common quantitative questions, and sql_query "
    "for anything else quantitative. Only answer using tool results or retrieved "
    "context -- never invent a number. When you state a number, say which tool or "
    "document it came from. Some data in this project (the hiring pipeline dates) "
    "is synthetic/simulated; say so if a question touches it."
)

SEARCH_DOCS_SPEC = ToolSpec(
    name="search_docs",
    description=(
        "Search this project's documentation (SQL findings, survival model "
        "findings, design decisions) for relevant context. Use for "
        "methodology or 'why' questions."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for"}},
        "required": ["query"],
    },
)

GET_ATTRITION_RATE_SPEC = ToolSpec(
    name="get_attrition_rate",
    description="Get headcount and attrition rate, optionally filtered by department and/or job role. Omit a parameter to not filter on it.",
    parameters={
        "type": "object",
        "properties": {
            "department": {
                "type": "string",
                "description": "e.g. 'Sales', 'Research & Development', 'Human Resources'",
            },
            "job_role": {
                "type": "string",
                "description": "e.g. 'Sales Representative', 'Research Scientist'",
            },
        },
    },
)

GET_FLIGHT_RISK_WATCHLIST_SPEC = ToolSpec(
    name="get_flight_risk_watchlist",
    description="Get the top N current employees ranked by predicted attrition hazard (from the survival model), optionally filtered by department.",
    parameters={
        "type": "object",
        "properties": {
            "department": {"type": "string"},
            "top_n": {"type": "integer", "description": f"How many employees to return, default {DEFAULT_WATCHLIST_TOP_N}"},
        },
    },
)

SQL_QUERY_SPEC = ToolSpec(
    name="sql_query",
    description=(
        "Run a read-only SQL SELECT query against the hr_analytics database for "
        "anything the other tools don't cover. Tables: employees (employee_number, "
        "age, attrition, department, job_role, monthly_income, over_time, "
        "years_at_company, job_satisfaction, ...) and hiring_pipeline "
        "(employee_number, time_to_fill_days, time_to_hire_days, ...). "
        "Results are capped at 50 rows."
    ),
    parameters={
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single SELECT or WITH...SELECT statement"}},
        "required": ["sql"],
    },
)

TOOLS: list[ToolSpec] = [SEARCH_DOCS_SPEC, GET_ATTRITION_RATE_SPEC, GET_FLIGHT_RISK_WATCHLIST_SPEC, SQL_QUERY_SPEC]


def _tool_get_attrition_rate(department: str | None = None, job_role: str | None = None) -> str:
    clauses, params = [], []
    if department:
        clauses.append("department = ?")
        params.append(department)
    if job_role:
        clauses.append("job_role = ?")
        params.append(job_role)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT COUNT(*) AS headcount, "
        "SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers, "
        "ROUND(100.0 * SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct "
        f"FROM employees {where}"
    )
    rows = sql_tool.execute_parameterized(sql, tuple(params))
    return json.dumps(rows)


def _tool_get_flight_risk_watchlist(department: str | None = None, top_n: int | str = DEFAULT_WATCHLIST_TOP_N) -> str:
    if not RISK_PATH.exists():
        return json.dumps({"error": f"{RISK_PATH} not found -- run the survival_model pipeline stage first."})
    # Tool-call arguments come from the model's JSON generation -- despite the
    # "integer" type in GET_FLIGHT_RISK_WATCHLIST_SPEC, not every provider/model
    # reliably emits a JSON number rather than a numeric string here. Coerce
    # explicitly rather than let min(top_n, MAX_ROWS) raise TypeError on a str.
    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = DEFAULT_WATCHLIST_TOP_N
    risk = pd.read_csv(RISK_PATH)
    current = risk[risk["Attrition"] == "No"]
    if department:
        current = current[current["Department"] == department]
    top = current.sort_values("predicted_hazard_score", ascending=False).head(max(1, min(top_n, sql_tool.MAX_ROWS)))
    records = top[["EmployeeNumber", "Department", "JobRole", "predicted_hazard_score"]].to_dict(orient="records")
    return json.dumps(records)


def _tool_sql_query(sql: str) -> str:
    rows = sql_tool.run_read_only_query(sql)
    return json.dumps(rows)


def _tool_search_docs(index: RetrievalIndex, query: str) -> tuple[str, list[tuple]]:
    results = retrieve(index, query)
    if not results:
        return "No relevant documentation found for that query.", []
    summary = "\n\n".join(f"[{c.doc_id} § {c.section}]\n{c.text}" for c, _ in results)
    return summary, results


def _execute_tool_call(index: RetrievalIndex, call: ToolCall) -> tuple[ToolResult, list[tuple]]:
    """Returns the ToolResult to feed back to the model, plus any newly
    retrieved chunks (only non-empty for search_docs) to accumulate into the
    turn's PromptSnapshot."""
    try:
        if call.name == "search_docs":
            content, chunks = _tool_search_docs(index, call.arguments.get("query", ""))
            return ToolResult(tool_call_id=call.id, content=truncate_tool_result(content)), chunks
        if call.name == "get_attrition_rate":
            content = _tool_get_attrition_rate(call.arguments.get("department"), call.arguments.get("job_role"))
            return ToolResult(tool_call_id=call.id, content=truncate_tool_result(content)), []
        if call.name == "get_flight_risk_watchlist":
            content = _tool_get_flight_risk_watchlist(
                call.arguments.get("department"), call.arguments.get("top_n", DEFAULT_WATCHLIST_TOP_N)
            )
            return ToolResult(tool_call_id=call.id, content=truncate_tool_result(content)), []
        if call.name == "sql_query":
            content = _tool_sql_query(call.arguments.get("sql", ""))
            return ToolResult(tool_call_id=call.id, content=truncate_tool_result(content)), []
        return ToolResult(tool_call_id=call.id, content=f"Unknown tool: {call.name}", is_error=True), []
    except Exception as exc:  # tool failures must feed back to the model as a
        # readable error, not crash the whole turn -- the model can retry with
        # different arguments or explain the failure to the user.
        return ToolResult(tool_call_id=call.id, content=f"Tool error: {exc}", is_error=True), []


def run_turn(
    provider: LLMProvider, index: RetrievalIndex, history: list[Message], user_input: str
) -> tuple[Message, PromptSnapshot]:
    messages = [*history, Message(role="user", content=user_input)]
    accumulated_chunks: list[tuple] = []
    snapshot: PromptSnapshot | None = None

    for _ in range(MAX_TOOL_ITERATIONS):
        snapshot = assemble_prompt(SYSTEM_PROMPT, messages, accumulated_chunks)
        response = provider.complete(system=snapshot.system, messages=snapshot.messages, tools=TOOLS)

        if response.stop_reason != "tool_use":
            return response.message, snapshot

        messages.append(response.message)
        tool_results = []
        for call in response.message.tool_calls:
            result, new_chunks = _execute_tool_call(index, call)
            tool_results.append(result)
            accumulated_chunks.extend(new_chunks)
        messages.append(Message(role="tool", tool_results=tuple(tool_results)))

    # Loop guard hit: return whatever the model last said rather than looping
    # forever, with an explicit note so this is visible, not a silent cutoff.
    fallback = Message(
        role="assistant",
        content="I wasn't able to finish answering within the tool-call limit for this turn. Try rephrasing or asking a narrower question.",
    )
    assert snapshot is not None
    return fallback, snapshot
