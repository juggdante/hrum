"""Safe arithmetic (course P4 `eval_aman`): no eval(), only numbers and + - * / ** and parentheses.

Every non-trivial input number must already exist in this run's evidence, so derived figures
(growth %, differences, shares) stay traceable to retrieved data.
"""
from __future__ import annotations

import ast
import operator
from typing import Any

from agent.core.context import RunContext
from agent.guardrails.grounding import evidence_numbers, extract_numbers, supported
from agent.tools import Tool

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** ( ) are allowed")

    return ev(ast.parse(expression.replace(",", "").replace("×", "*"), mode="eval"))


UNIT_CONSTANTS = [100.0, 1000.0, 1e6, 1e9, 12.0, 52.0, 365.0, 365.25, 2080.0]  # %, scales, calendar, FTE hours


def _calculate(args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    expr = args["expression"]
    known = UNIT_CONSTANTS + [v for e in ctx.ledger.values() for v in evidence_numbers(e.result)]
    unknown = [n.raw for n in extract_numbers(expr.replace(",", "")) if not n.trivial and not supported(n, known)]
    if unknown:
        raise ValueError(f"input number(s) {unknown[:3]} are not in retrieved evidence; query them first")
    return {"expression": expr, "result": safe_eval(expr), "label": args.get("label", "")}


CALCULATE = Tool(
    name="calculate",
    description="Evaluate arithmetic on numbers already retrieved (growth %, differences, shares). "
                "Example: (7523952094.89 / 6918855698.17 - 1) * 100",
    parameters={"type": "object",
                "properties": {"expression": {"type": "string"},
                               "label": {"type": "string", "description": "What the result means"}},
                "required": ["expression"], "additionalProperties": False},
    func=_calculate, roles=frozenset({"chro", "analyst"}),
)
