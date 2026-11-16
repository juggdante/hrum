"""Documentation search tool (RAG over docs/, the FAQ and metric definitions)."""
from __future__ import annotations

from typing import Any

from agent.core.context import RunContext
from agent.tools import Tool


def _search_docs(args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from agent.rag.retrieve import retriever

    hits = retriever().search(args["query"], k=args.get("k", 4))
    removed = sum(h["lines_removed"] for h in hits)
    out: dict[str, Any] = {"passages": [{k: h[k] for k in ("chunk_id", "source", "section", "text")} for h in hits]}
    if removed:
        out["note"] = f"{removed} line(s) that looked like instructions were removed from the documents"
    return out


SEARCH_DOCS = Tool(
    name="search_docs",
    description="Search HRUM documentation: metric definitions, data sources and limitations, what is real vs "
                "synthetic, privacy rules, how the attrition model works. Use for 'what/why/how' questions.",
    parameters={"type": "object",
                "properties": {"query": {"type": "string"},
                               "k": {"type": "integer", "minimum": 1, "maximum": 8}},
                "required": ["query"], "additionalProperties": False},
    func=_search_docs, roles=frozenset({"chro", "analyst"}),
)
