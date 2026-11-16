"""Free-form read-only SQL over `mart` (analyst role only), protected by agent/guardrails/sql_guard.py."""
from __future__ import annotations

import threading
from typing import Any

import duckdb

from agent.core.context import RunContext
from agent.guardrails.sql_guard import check_sql
from agent.tools import Tool
from common.config import load_config


def mart_schema_text(ctx: RunContext) -> str:
    rows = ctx.connection().execute(
        "select table_name, string_agg(column_name || ' ' || data_type, ', ' order by ordinal_position) "
        "from information_schema.columns where table_schema = 'mart' group by 1 order by 1").fetchall()
    blocked = set(load_config("guardrails")["privacy"]["blocked_columns"])
    return "\n".join(f"mart.{t}({', '.join(c for c in cols.split(', ') if c.split()[0] not in blocked)})"
                     for t, cols in rows)


def execute_with_timeout(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple]]:
    """Run already-guarded SQL; interrupt it after guardrails.sql.timeout_seconds. Shared with the app."""
    timer = threading.Timer(float(load_config("guardrails")["sql"]["timeout_seconds"]), con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        return [c[0] for c in cur.description], cur.fetchall()
    finally:
        timer.cancel()


def _run_sql(args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    guarded = check_sql(args["sql"])
    cols, data = execute_with_timeout(ctx.connection(), guarded.sql)
    rows = [dict(zip(cols, r)) for r in data]
    min_group = load_config("guardrails")["privacy"]["min_group_size"]
    caveats = [f"Groups with fewer than {min_group} records are suppressed for privacy."]
    if "mart.fct_employee_snapshot" in guarded.tables:
        caveats.insert(0, "Uses SYNTHETIC HRIS fields: demonstrates capability, not a real finding about SF.")
    return {"rows": rows, "row_count": len(rows), "sql": guarded.sql, "caveats": caveats}


RUN_SQL = Tool(
    name="run_sql",
    description="Run ONE read-only aggregate SELECT (DuckDB dialect) on mart.* tables when no governed metric "
                "fits. Outer SELECT must aggregate; person-level keys are blocked; small groups are suppressed.",
    parameters={"type": "object",
                "properties": {"sql": {"type": "string", "description": "A single DuckDB SELECT over mart.*"}},
                "required": ["sql"], "additionalProperties": False},
    func=_run_sql, roles=frozenset({"analyst"}),
)
