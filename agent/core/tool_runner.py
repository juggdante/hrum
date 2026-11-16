"""One door for every tool call (course P4 `jalankan_tool` + `normalisasi_argumen`).

Liberal on input (small models send lists as JSON strings or single values for arrays), strict after:
JSON-schema validation, role check, duplicate-call cache (loop detection), evidence ledger, trace.
"""
from __future__ import annotations

import json
import time
from typing import Any

import jsonschema

from agent.core.context import RunContext
from agent.observability import tracing
from agent.tools import Tool, registry


def normalize_arguments(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    props = tool.parameters.get("properties", {})
    for key, value in list(args.items()):
        spec = props.get(key, {})
        if spec.get("type") == "array":
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            if not isinstance(value, list):
                value = [value]
            args[key] = [v.strip().lower() if isinstance(v, str) else v for v in value]
        elif spec.get("type") == "object" and isinstance(value, str):
            try:
                args[key] = json.loads(value) if value.strip() else {}
            except json.JSONDecodeError:
                pass
        elif spec.get("type") == "integer" and isinstance(value, str) and value.strip().isdigit():
            args[key] = int(value)
        if value is None or value == "" or value == {}:
            args.pop(key, None)
    return args


def _parse(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise TypeError("arguments must be a JSON object")
    return parsed


def run_tool(name: str, raw_arguments: str | dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    """Execute a tool call and return the message payload for the model (never raises). Traced as one span."""
    as_type = "retriever" if name == "search_docs" else "tool"
    with tracing.observation(f"tool:{name}", as_type=as_type, input=raw_arguments) as span:
        payload = _run_tool(name, raw_arguments, ctx)
        span.update(output=to_model_text(payload, 4000),
                    metadata={"evidence_id": payload.get("evidence_id"), "role": ctx.role},
                    level="WARNING" if "error" in payload else "DEFAULT",
                    status_message=str(payload.get("error", ""))[:300] or None)
    return payload


def _run_tool(name: str, raw_arguments: str | dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    t0 = time.time()
    tools = registry()
    tool = tools.get(name)
    if tool is None or ctx.role not in tool.roles:
        ctx.log("tool_denied", tool=name, role=ctx.role)
        return {"error": f"tool '{name}' is not available for role '{ctx.role}'"}
    try:
        args = normalize_arguments(tool, _parse(raw_arguments))
        jsonschema.validate(args, tool.parameters)
    except Exception as e:  # noqa: BLE001 - message goes back to the model
        n_err = 1 + sum(1 for ev in ctx.trace if ev["kind"] == "tool_error" and ev.get("tool") == name)
        ctx.log("tool_error", tool=name, arguments=str(raw_arguments)[:300], error=str(e)[:300])
        msg = f"invalid arguments for {name}: {str(e).splitlines()[0][:300]}"
        if n_err >= 2:
            msg += f" (error #{n_err}). Expected schema: {json.dumps(tool.parameters)[:800]}"
        return {"error": msg}

    key = name + json.dumps(args, sort_keys=True, default=str)
    if key in ctx.cache:
        ctx.log("tool_repeat", tool=name, arguments=args)
        return {"note": "identical call already made; reuse its result", **ctx.cache[key]}

    ctx.meter.tool_calls += 1
    try:
        result = tool.func(args, ctx)
    except Exception as e:  # noqa: BLE001
        ctx.log("tool_failed", tool=name, arguments=args, error=f"{type(e).__name__}: {str(e)[:300]}")
        return {"error": f"{type(e).__name__}: {str(e)[:400]}"}

    payload: dict[str, Any] = dict(result)
    if tool.is_evidence:
        payload = {"evidence_id": ctx.add_evidence(name, args, result), **result}
    ctx.cache[key] = payload
    ctx.log("tool", tool=name, arguments=args, seconds=round(time.time() - t0, 3),
            evidence_id=payload.get("evidence_id"), rows=result.get("row_count"), sql=result.get("sql"))
    return payload


def to_model_text(payload: dict[str, Any], max_chars: int = 8000) -> str:
    """Serialize a tool result for the model: rounded floats, truncated."""
    def rnd(o: Any) -> Any:
        if isinstance(o, float):
            return round(o, 4)
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items() if k != "params"}
        if isinstance(o, list):
            return [rnd(v) for v in o]
        return o

    return json.dumps(rnd(payload), ensure_ascii=False, default=str)[:max_chars]
