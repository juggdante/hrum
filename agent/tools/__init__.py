"""Tool registry. Each tool: JSON schema for the LLM, the roles allowed to use it, and a Python function.

Tool functions take (arguments, ctx) and return a JSON-able dict. Results that contain facts are added
to the evidence ledger by the tool runner and receive an evidence id (E1, E2, ...).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.core.context import RunContext


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[dict[str, Any], RunContext], dict[str, Any]]
    roles: frozenset[str]
    is_evidence: bool = True     # results go into the ledger and can be cited

    def schema(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def registry() -> dict[str, Tool]:
    from agent.tools.attrition import ATTRITION_INSIGHT
    from agent.tools.calc import CALCULATE
    from agent.tools.docs import SEARCH_DOCS
    from agent.tools.metrics import LIST_METRICS, QUERY_METRIC
    from agent.tools.sql import RUN_SQL

    tools = [LIST_METRICS, QUERY_METRIC, RUN_SQL, SEARCH_DOCS, ATTRITION_INSIGHT, CALCULATE]
    return {t.name: t for t in tools}


def tools_for_role(role: str) -> dict[str, Tool]:
    return {n: t for n, t in registry().items() if role in t.roles}
