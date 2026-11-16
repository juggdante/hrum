"""Semantic-layer tools: the default, safest way for the agent to get numbers."""
from __future__ import annotations

from typing import Any

from agent.core.context import RunContext
from agent.tools import Tool
from common.semantic import catalog, compile_query, describe_catalog, run_query

ALL_ROLES = frozenset({"chro", "analyst"})


def _list_metrics(args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    return describe_catalog()


def _query_metric(args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    q = compile_query(metrics=args["metrics"], dimensions=args.get("dimensions"), filters=args.get("filters"),
                      order_by=args.get("order_by"), descending=args.get("descending", True),
                      limit=args.get("limit", 20))
    return run_query(ctx.connection(), q)


LIST_METRICS = Tool(
    name="list_metrics",
    description="List every governed HR metric (with definition, real vs synthetic) and the dimensions "
                "it can be grouped or filtered by.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    func=_list_metrics, roles=ALL_ROLES, is_evidence=False,
)


def _query_metric_schema() -> dict[str, Any]:
    cat = catalog()
    dims = sorted({d for m in cat["models"].values() for d in m["dimensions"]})
    return {
        "type": "object",
        "properties": {
            "metrics": {"type": "array", "items": {"type": "string", "enum": sorted(cat["metrics"])},
                        "minItems": 1, "description": "Metrics from the same model (see list_metrics)."},
            "dimensions": {"type": "array", "items": {"type": "string", "enum": dims},
                           "description": "Group-by dimensions, e.g. ['fiscal_year'] or ['department']."},
            "filters": {"type": "object",
                        "description": "dimension -> value, list of values, or {gte/lte/gt/lt}. "
                                       "Example: {\"fiscal_year\": 2026} or {\"fiscal_year\": {\"gte\": 2020}}."},
            "order_by": {"type": "string", "description": "A requested metric or dimension."},
            "descending": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": ["metrics"],
        "additionalProperties": False,
    }


QUERY_METRIC = Tool(
    name="query_metric",
    description="Compute governed HR metrics grouped by dimensions with filters. Returns rows, the SQL that "
                "ran, metric definitions and caveats. Prefer this over run_sql.",
    parameters=_query_metric_schema(),
    func=_query_metric, roles=ALL_ROLES,
)
