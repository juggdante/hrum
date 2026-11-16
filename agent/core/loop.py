"""HRUM agent: tool-calling loop with budgets -> structured final answer -> grounding enforcement.

Adapted from course lab P4 (`jawab_agent`, `finalisasi`, `tegakkan_grounding`):
  1. loop: the model calls tools until it writes a draft, or a step/tool/time budget runs out
  2. finalize: the model turns evidence + draft into JSON with claims that cite evidence ids
  3. grounding: validate -> one repair -> drop unsupported claims -> abstain if nothing is left
Caveats and SQL are attached by code from the evidence (CLAUDE.md rule 8), not written by the model.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from agent.core.context import RunContext
from agent.core.tool_runner import run_tool, to_model_text
from agent.guardrails.grounding import Validation, validate
from agent.guardrails.number_style import normalize_answer
from agent.llm.base import LLMClient, LLMResponse
from agent.observability import tracing
from agent.prompts import FINALIZE, PROMPT_VERSION, REPAIR, SQL_RULE_ANALYST, SQL_RULE_CHRO, SYSTEM
from agent.tools import tools_for_role
from common.config import load_config
from common.semantic import describe_catalog

ABSTAIN_TEXT = "The available evidence is not enough to answer with verifiable numbers."


@dataclass
class AgentResult:
    question: str
    role: str
    status: str
    answer: str
    claims: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    caveats: list[str]
    grounding: dict[str, Any]
    repair_level: int          # 0 valid first time, 1 repaired, 2 claims dropped, 3 abstained
    stop_reason: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_schema(evidence_ids: list[str]) -> dict[str, Any]:
    ids = {"type": "string", "enum": evidence_ids} if evidence_ids else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["answered", "abstain", "refused"]},
            "answer": {"type": "string"},
            "claims": {"type": "array", "maxItems": 8 if evidence_ids else 0,
                       "items": {"type": "object",
                                 "properties": {"text": {"type": "string"},
                                                "evidence_ids": {"type": "array", "items": ids, "minItems": 1}},
                                 "required": ["text", "evidence_ids"]}},
        },
        "required": ["status", "answer", "claims"],
    }


def _metrics_text() -> str:
    cat = describe_catalog()["metrics"]
    return "\n".join(f"- {n}{' [SYNTHETIC]' if m['synthetic'] else ''}: {m['description']}" for n, m in cat.items())


def system_prompt(ctx: RunContext) -> str:
    if ctx.role == "analyst":
        from agent.tools.sql import mart_schema_text

        sql_rule = SQL_RULE_ANALYST.format(schema=mart_schema_text(ctx))
    else:
        sql_rule = SQL_RULE_CHRO
    return SYSTEM.format(today=date.today().isoformat(), role=ctx.role, metrics=_metrics_text(), sql_rule=sql_rule)


def _call(llm: LLMClient, ctx: RunContext, step: str, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
    t0 = time.time()
    with tracing.observation(step, as_type="generation", model=getattr(llm, "model", None), input=messages,
                             model_parameters={"max_tokens": kw.get("max_tokens", 800),
                                               "tools": len(kw.get("tools") or []),
                                               "json_schema": bool(kw.get("json_schema"))}) as gen:
        r = llm.chat(messages, **kw)
        gen.update(output={"text": r.text, "tool_calls": [{"name": c.name, "arguments": c.arguments}
                                                          for c in r.tool_calls]},
                   usage_details={"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens})
    dt = time.time() - t0
    ctx.meter.llm_calls += 1
    ctx.meter.prompt_tokens += r.usage.prompt_tokens
    ctx.meter.completion_tokens += r.usage.completion_tokens
    ctx.meter.llm_seconds += dt
    ctx.log("llm", step=step, seconds=round(dt, 3), tool_calls=[c.name for c in r.tool_calls],
            prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
            text=r.text[:300])
    return r


def _evidence_text(ctx: RunContext) -> str:
    if not ctx.ledger:
        return "(no evidence was retrieved)"
    return "\n".join(f"[{e.id}] {e.tool}({json.dumps(e.arguments, default=str)}) -> {to_model_text(e.result, 3000)}"
                     for e in ctx.ledger.values())


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def run_loop(question: str, llm: LLMClient, ctx: RunContext,
             history: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    """Tool-calling phase. Returns (draft_text, stop_reason)."""
    budget = load_config("guardrails")["agent"]
    tools = tools_for_role(ctx.role)
    schemas = [t.schema() for t in tools.values()]
    messages = [{"role": "system", "content": system_prompt(ctx)}, *(history or []),
                {"role": "user", "content": question}]
    n_tools = 0
    for step in range(budget["max_steps"]):
        if time.time() - ctx.started > budget["max_seconds"]:
            return "", "time_budget"
        r = _call(llm, ctx, f"step_{step}", messages, tools=schemas, max_tokens=700)
        if not r.tool_calls:
            return r.text, "done"
        messages.append({"role": "assistant", "content": r.text or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.name, "arguments": c.arguments}}
                                        for c in r.tool_calls]})
        for c in r.tool_calls:
            if n_tools >= budget["max_tool_calls"]:
                payload = {"error": "tool budget exhausted; answer from the evidence you have"}
            else:
                payload = run_tool(c.name, c.arguments, ctx)
                n_tools += 1
            messages.append({"role": "tool", "tool_call_id": c.id, "content": to_model_text(payload)})
        if n_tools >= budget["max_tool_calls"]:
            return "", "tool_budget"
    return "", "step_budget"


def finalize(question: str, draft: str, llm: LLMClient, ctx: RunContext) -> dict[str, Any]:
    prompt = FINALIZE.format(question=question, evidence=_evidence_text(ctx), draft=draft or "(none)")
    r = _call(llm, ctx, "finalize", [{"role": "user", "content": prompt}],
              json_schema=answer_schema(list(ctx.ledger)), max_tokens=900)
    try:
        return _parse_json(r.text)
    except (json.JSONDecodeError, ValueError):
        ctx.log("finalize_unparsable", text=r.text[:300])
        return {"status": "abstain", "answer": draft or ABSTAIN_TEXT, "claims": []}


def enforce_grounding(question: str, answer: dict[str, Any], llm: LLMClient,
                      ctx: RunContext) -> tuple[dict[str, Any], Validation, int]:
    ledger = ctx.evidence_payloads()
    with tracing.observation("grounding", as_type="guardrail", input={"answer": answer.get("answer"),
                                                                      "claims": answer.get("claims")}) as g:
        v = validate(answer, ledger)
        g.update(output={"ok": v.ok, "claims_ok": v.claims_ok, "claims_total": v.claims_total,
                         "problems": v.problems}, level="DEFAULT" if v.ok else "WARNING")
    ctx.log("grounding", ok=v.ok, problems=v.problems)
    if v.ok:
        return answer, v, 0
    prompt = REPAIR.format(question=question, evidence=_evidence_text(ctx),
                           answer=json.dumps(answer, ensure_ascii=False), problems="\n- ".join(v.problems))
    try:
        r = _call(llm, ctx, "repair", [{"role": "user", "content": prompt}],
                  json_schema=answer_schema(list(ctx.ledger)), max_tokens=900)
        repaired = _parse_json(r.text)
    except (json.JSONDecodeError, ValueError):
        repaired = answer
    v2 = validate(repaired, ledger)
    ctx.log("grounding_repair", ok=v2.ok, problems=v2.problems)
    if v2.ok:
        return repaired, v2, 1
    kept = [c for c in repaired.get("claims", [])
            if validate({"status": "answered", "answer": "", "claims": [c]}, ledger).ok]
    if kept:
        trimmed = {"status": "answered", "answer": " ".join(c["text"] for c in kept), "claims": kept}
        return trimmed, validate(trimmed, ledger), 2
    return {"status": "abstain", "answer": ABSTAIN_TEXT, "claims": []}, v2, 3


def auto_retrieve(question: str, ctx: RunContext, stop_reason: str) -> None:
    """Safety net: if the model stopped without calling any tool, search the docs with the question itself.

    Small models sometimes answer 'why/how' questions with no tool call and then abstain for lack of evidence.
    Finalisation still decides the status, so refusals and genuine abstains are unaffected.
    """
    if ctx.ledger or stop_reason != "done" or "search_docs" not in tools_for_role(ctx.role):
        return
    ctx.log("auto_retrieve", query=question)
    run_tool("search_docs", {"query": question}, ctx)


def dedupe_citations(answer: dict[str, Any]) -> dict[str, Any]:
    """Small models sometimes cite the same evidence twice in one claim (["E1", "E1"])."""
    for c in answer.get("claims", []):
        c["evidence_ids"] = list(dict.fromkeys(c.get("evidence_ids", [])))
    return answer


def answer_question(question: str, llm: LLMClient, role: str = "chro",
                    history: list[dict[str, Any]] | None = None, ctx: RunContext | None = None,
                    trace: dict[str, Any] | None = None) -> AgentResult:
    """Answer one question. `trace` adds Langfuse attributes: session_id, tags, metadata (all optional)."""
    ctx = ctx or RunContext(role=role)
    trace = trace or {}
    with tracing.trace_attributes(trace_name="hrum.answer", user_id=f"role:{ctx.role}",
                                  session_id=trace.get("session_id"),
                                  tags=[f"role:{ctx.role}", f"prompt:{PROMPT_VERSION}", *trace.get("tags", [])],
                                  metadata={k: str(v) for k, v in trace.get("metadata", {}).items()} or None), \
         tracing.observation("answer", as_type="agent", input=question,
                             metadata={"model": getattr(llm, "model", ""), "role": ctx.role}) as root:
        result = _answer(question, llm, ctx, history)
        root.update(output={"status": result.status, "answer": result.answer, "claims": result.claims,
                            "caveats": result.caveats},
                    metadata={"grounding": result.grounding, "repair_level": result.repair_level,
                              "stop_reason": result.stop_reason, "usage": result.usage,
                              "evidence": [{"id": e["id"], "tool": e["tool"], "cited": e["cited"]}
                                           for e in result.evidence]},
                    level="DEFAULT" if result.grounding.get("ok") else "WARNING")
        root.score_trace(name="grounding_ok", value=1 if result.grounding.get("ok") else 0, data_type="BOOLEAN")
        root.score_trace(name="repair_level", value=result.repair_level, data_type="NUMERIC")
        root.score_trace(name="status", value=result.status, data_type="CATEGORICAL")
    return result


def _answer(question: str, llm: LLMClient, ctx: RunContext,
            history: list[dict[str, Any]] | None) -> AgentResult:
    ctx.log("question", question=question, role=ctx.role, model=getattr(llm, "model", ""))
    try:
        draft, stop_reason = run_loop(question, llm, ctx, history)
        ctx.log("stop", reason=stop_reason, tools=ctx.meter.tool_calls)
        auto_retrieve(question, ctx, stop_reason)
        final, v, level = enforce_grounding(question, finalize(question, draft, llm, ctx), llm, ctx)
        final = normalize_answer(dedupe_citations(final))   # after grounding: rewrites keep the value
    finally:
        ctx.close()
    cited = {i for c in final.get("claims", []) for i in c.get("evidence_ids", [])}
    evidence = [{"id": e.id, "tool": e.tool, "arguments": e.arguments, "sql": e.result.get("sql"),
                 "definitions": e.result.get("definitions"), "rows": e.result.get("rows"),
                 "result": e.result.get("result"), "cited": e.id in cited}
                for e in ctx.ledger.values()]
    caveats = list(dict.fromkeys(c for e in ctx.ledger.values() if e.id in cited
                                 for c in e.result.get("caveats", [])))
    return AgentResult(question=question, role=ctx.role, status=final.get("status", "abstain"),
                       answer=final.get("answer", ""), claims=final.get("claims", []), evidence=evidence,
                       caveats=caveats, grounding=asdict(v), repair_level=level, stop_reason=stop_reason,
                       trace=ctx.trace, usage={**asdict(ctx.meter), "seconds": round(time.time() - ctx.started, 2)})
