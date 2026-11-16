"""Compile semantic-layer requests (metrics x dimensions x filters) into safe DuckDB SQL.

The metric catalogue lives in warehouse/semantic/metrics.yaml (one definition per metric).
Values from the caller are always bound as parameters, never concatenated into SQL.
Groups smaller than guardrails.privacy.min_group_size are suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import duckdb
import yaml

from common.config import ROOT, load_config

SEMANTIC_PATH = ROOT / "warehouse" / "semantic" / "metrics.yaml"
MAX_ROWS = 200


class SemanticError(ValueError):
    """Raised for unknown metrics/dimensions or invalid filters. Message is shown to the LLM."""


@lru_cache
def catalog() -> dict[str, Any]:
    with open(SEMANTIC_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def min_group_size() -> int:
    return int(load_config("guardrails")["privacy"]["min_group_size"])


@dataclass
class MetricQuery:
    sql: str
    params: dict[str, Any]
    model: str
    metrics: list[str]
    dimensions: list[str]
    caveats: list[str] = field(default_factory=list)


def describe_catalog() -> dict[str, Any]:
    """Compact catalogue for the LLM: metrics with descriptions and the dimensions they support."""
    cat = catalog()
    return {
        "metrics": {name: {"description": m["description"], "model": m["model"],
                           "synthetic": bool(m.get("synthetic") or cat["models"][m["model"]]["is_synthetic"])}
                    for name, m in cat["metrics"].items()},
        "dimensions_by_model": {name: {d: v["description"] for d, v in mdl["dimensions"].items()}
                                for name, mdl in cat["models"].items()},
    }


def _filter_clause(dim_sql: str, key: str, value: Any, params: dict[str, Any]) -> str:
    if isinstance(value, dict):
        parts = []
        for op, sym in (("gte", ">="), ("lte", "<="), ("gt", ">"), ("lt", "<")):
            if op in value:
                p = f"p{len(params)}"
                params[p] = value[op]
                parts.append(f"{dim_sql} {sym} ${p}")
        if not parts:
            raise SemanticError(f"filter '{key}': use a value, a list, or {{gte/lte/gt/lt}}")
        return " and ".join(parts)
    if isinstance(value, list):
        if not value:
            raise SemanticError(f"filter '{key}': empty list")
        names = []
        for v in value:
            p = f"p{len(params)}"
            params[p] = v
            names.append(f"${p}")
        return f"{dim_sql} in ({', '.join(names)})"
    p = f"p{len(params)}"
    params[p] = value
    return f"{dim_sql} = ${p}"


def compile_query(metrics: list[str], dimensions: list[str] | None = None,
                  filters: dict[str, Any] | None = None, order_by: str | None = None,
                  descending: bool = True, limit: int = 50) -> MetricQuery:
    cat = catalog()
    dimensions, filters = list(dimensions or []), dict(filters or {})
    if not metrics:
        raise SemanticError("at least one metric is required")
    unknown = [m for m in metrics if m not in cat["metrics"]]
    if unknown:
        raise SemanticError(f"unknown metric(s) {unknown}; valid: {sorted(cat['metrics'])}")
    models = {cat["metrics"][m]["model"] for m in metrics}
    if len(models) > 1:
        raise SemanticError(f"metrics come from different models {sorted(models)}; query them separately")
    model_name = models.pop()
    model = cat["models"][model_name]
    dims = model["dimensions"]
    bad = [d for d in dimensions + list(filters) if d not in dims]
    if bad:
        raise SemanticError(f"dimension(s) {bad} not available for {model_name}; valid: {sorted(dims)}")
    if order_by and order_by not in metrics + dimensions:
        raise SemanticError("order_by must be one of the requested metrics or dimensions")

    params: dict[str, Any] = {}
    select = [f"{dims[d]['sql']} as {d}" for d in dimensions]
    select += [f"{cat['metrics'][m]['sql']} as {m}" for m in metrics]
    select.append("count(*) as group_size")
    where = [model["default_filter"]] + [_filter_clause(dims[k]["sql"], k, v, params) for k, v in filters.items()]
    sql = f"select {', '.join(select)}\nfrom {model['table']} {model['alias']}\n"
    sql += "\n".join(model.get("joins", [])) + f"\nwhere {' and '.join(f'({w})' for w in where)}"
    if dimensions:
        sql += f"\ngroup by {', '.join(dims[d]['sql'] for d in dimensions)}"
    sql += f"\nhaving count(*) >= {min_group_size()}"
    order = order_by or (dimensions[0] if dimensions and dimensions[0] == "fiscal_year" else metrics[0])
    sql += f"\norder by {order} {'desc' if descending and order != 'fiscal_year' else 'asc'} nulls last"
    sql += f"\nlimit {max(1, min(int(limit), MAX_ROWS))}"

    caveats = []
    if model["is_synthetic"] or any(cat["metrics"][m].get("synthetic") for m in metrics):
        caveats.append("Uses SYNTHETIC HRIS fields: demonstrates capability, not a real finding about SF.")
    if model_name == "compensation":
        caveats.append("Fiscal years (July-June). Counts pay records, not unique people.")
        if {"employment_type", "is_full_time"} & set(dimensions + list(filters)):
            caveats.append("employment_type / hours are only populated from FY2017.")
    caveats.append(f"Groups with fewer than {min_group_size()} records are suppressed for privacy.")
    return MetricQuery(sql, params, model_name, metrics, dimensions, caveats)


def run_query(con: duckdb.DuckDBPyConnection, q: MetricQuery) -> dict[str, Any]:
    cur = con.execute(q.sql, q.params)
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cat = catalog()
    return {
        "rows": rows,
        "row_count": len(rows),
        "sql": q.sql,
        "params": q.params,
        "definitions": {m: cat["metrics"][m]["description"] for m in q.metrics},
        "caveats": q.caveats,
    }
