"""Ask HRUM from the terminal.

    python -m agent.cli "Berapa total kompensasi per tahun sejak FY2020?" --role chro
    python -m agent.cli --smoke          # one tool-calling round trip against the configured LLM
"""
from __future__ import annotations

import argparse
import json

from agent.core.loop import AgentResult, answer_question
from agent.llm.base import get_client
from agent.observability import tracing


def render(r: AgentResult) -> str:
    lines = [f"[{r.status}] {r.answer}", ""]
    for i, c in enumerate(r.claims, 1):
        lines.append(f"  claim {i}: {c['text']}  <- {', '.join(c['evidence_ids'])}")
    for e in r.evidence:
        mark = "*" if e["cited"] else " "
        lines.append(f"{mark} {e['id']} {e['tool']} {json.dumps(e['arguments'], default=str)}")
        if e.get("sql"):
            lines.append("    " + e["sql"].replace("\n", "\n    "))
    if r.caveats:
        lines += ["", "Caveats:"] + [f"  - {c}" for c in r.caveats]
    u = r.usage
    lines += ["", (f"grounding ok={r.grounding['ok']} repair_level={r.repair_level} stop={r.stop_reason} "
                  f"llm_calls={u['llm_calls']} tools={u['tool_calls']} tokens={u['prompt_tokens']}+"
                  f"{u['completion_tokens']} seconds={u['seconds']}")]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--role", default="chro", choices=["chro", "analyst"])
    ap.add_argument("--provider", default=None, help="provider name in configs/llm.yaml")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    question = "Berapa total kompensasi FY2026 dan headcount-nya?" if args.smoke else args.question
    if not question:
        ap.error("question is required")
    r = answer_question(question, get_client(args.provider), role=args.role,
                        trace={"tags": ["cli", "smoke"] if args.smoke else ["cli"]})
    print(json.dumps(r.to_dict(), indent=2, default=str) if args.json else render(r))
    if tracing.enabled():
        tracing.flush()   # the process exits right after; send buffered spans first
        print("trace: sent to Langfuse")


if __name__ == "__main__":
    main()
