"""Shared, cached data access for the Streamlit pages (read-only DuckDB, catalogue, docs, eval results)."""
from __future__ import annotations

import json
import re
from typing import Any

import duckdb
import pandas as pd
import streamlit as st
import yaml

from common.config import ROOT, db_path, load_config
from common.semantic import catalog as semantic_catalog


@st.cache_resource
def _connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path()), read_only=True)


def cursor() -> duckdb.DuckDBPyConnection:
    """A cursor per use: the cached connection is shared across sessions, cursors are not."""
    return _connection().cursor()


def query_df(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    return cursor().execute(sql, params or {}).df()


@st.cache_data(ttl=600)
def catalog() -> dict[str, Any]:
    return semantic_catalog()


@st.cache_data(ttl=600)
def mart_schema() -> pd.DataFrame:
    """table, column, type, description, blocked: one row per mart column (descriptions from dbt YAML)."""
    df = query_df("select table_name as \"table\", column_name as \"column\", data_type as type "
                  "from information_schema.columns where table_schema = 'mart' "
                  "order by table_name, ordinal_position")
    desc = _dbt_descriptions()
    blocked = {c.lower() for c in load_config("guardrails")["privacy"]["blocked_columns"]}
    df["description"] = [desc["columns"].get((t, c), "") for t, c in zip(df["table"], df["column"])]
    df["blocked"] = df["column"].str.lower().isin(blocked)
    return df


@st.cache_data(ttl=600)
def mart_columns() -> dict[str, set[str]]:
    """'mart.<table>' -> column names, for the SQL guard's SELECT * expansion."""
    out: dict[str, set[str]] = {}
    for t, c in zip(mart_schema()["table"], mart_schema()["column"]):
        out.setdefault(f"mart.{t}", set()).add(c.lower())
    return out


@st.cache_data(ttl=600)
def table_descriptions() -> dict[str, str]:
    return _dbt_descriptions()["tables"]


def _dbt_descriptions() -> dict[str, Any]:
    tables: dict[str, str] = {}
    columns: dict[tuple[str, str], str] = {}
    for path in (ROOT / "warehouse" / "models" / "marts").glob("*.yml"):
        for model in (yaml.safe_load(path.read_text()) or {}).get("models", []):
            tables[model["name"]] = " ".join(str(model.get("description", "")).split())
            for col in model.get("columns", []):
                if col.get("description"):
                    columns[(model["name"], col["name"])] = " ".join(str(col["description"]).split())
    return {"tables": tables, "columns": columns}


@st.cache_data(ttl=600)
def faq_sections() -> list[tuple[str, str]]:
    """(heading, body) per '## ' section of docs/kb/hrum_faq.md (the knowledge base the agent also uses)."""
    text = (ROOT / "docs" / "kb" / "hrum_faq.md").read_text()
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    return [(p.split("\n", 1)[0].strip(), p.split("\n", 1)[1].strip() if "\n" in p else "") for p in parts]


@st.cache_data(ttl=600)
def golden_questions() -> list[dict[str, Any]]:
    from evals.checks import load_questions

    return load_questions()


@st.cache_data(ttl=600)
def eval_results() -> dict[str, Any]:
    """Latest committed eval report of the shipped model (empty when absent)."""
    path = ROOT / "evals" / "reports" / "qwen3.5-4b.json"
    return json.loads(path.read_text()) if path.exists() else {}


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit or no browser context: assume light
        return False
