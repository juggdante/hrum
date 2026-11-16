"""Release gate: absolute thresholds + paired McNemar test vs the baseline (course lab P10 `gerbang.py`).

    python -m evals.gate --candidate evals/reports/qwen3.5-9b.json --baseline evals/reports/qwen3.5-4b.json
    python -m evals.gate --candidate evals/reports/qwen3.5-4b.json          # thresholds only

Decision
  REJECT         candidate misses a threshold, or is significantly worse than the baseline
  PROMOTE        candidate passes and is significantly better (or the baseline itself fails the gate)
  KEEP_BASELINE  candidate passes but the difference is not significant: keep the smaller, cheaper model
  PASS / FAIL    single-model mode (no baseline)
Exit code 1 for REJECT / FAIL. Writes evals/reports/gate.json and gate.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.config import ROOT
from evals.stats import mcnemar_exact

REPORTS = ROOT / "evals" / "reports"
THRESHOLDS = ROOT / "evals" / "thresholds.json"


def threshold_checks(r: dict[str, Any], t: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("accuracy", r["accuracy"]["rate"], ">=", t["min_accuracy"]),
        ("attack breaches", r["attack_breach"]["k"], "<=", t["max_attack_breach"]),
        ("grounding ok when answered", r["grounding_ok_when_answered"]["rate"], ">=",
         t["min_grounding_ok_when_answered"]),
        ("errors", r["errors"], "<=", t["max_errors"]),
        ("p95 latency (s)", r["latency_s"]["p95"], "<=", t["max_p95_latency_s"]),
    ]
    def ok(v: float | None, op: str, lim: float) -> bool:
        return True if v is None else (v >= lim if op == ">=" else v <= lim)   # None = not applicable

    return [{"check": name, "value": "n/a" if v is None else v, "op": op, "limit": lim, "ok": ok(v, op, lim)}
            for name, v, op, lim in rows]


def compare(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    shared = sorted(set(base["per_question"]) & set(cand["per_question"]))
    b = [q for q in shared if base["per_question"][q] and not cand["per_question"][q]]
    c = [q for q in shared if cand["per_question"][q] and not base["per_question"][q]]
    return {"paired_questions": len(shared), "only_baseline_passed": b, "only_candidate_passed": c,
            "p_value": round(mcnemar_exact(len(b), len(c)), 4),
            "same_question_set": base.get("questions_sha") == cand.get("questions_sha"),
            "same_prompt": base.get("prompt_version") == cand.get("prompt_version")}


def decide(cand_ok: bool, base_ok: bool | None, cmp: dict[str, Any] | None, alpha: float) -> tuple[str, str]:
    if cmp is None:
        return ("PASS", "all thresholds met") if cand_ok else ("FAIL", "threshold missed")
    better = len(cmp["only_candidate_passed"]) > len(cmp["only_baseline_passed"])
    significant = cmp["p_value"] < alpha
    if not cand_ok:
        return "REJECT", "candidate misses a threshold"
    if significant and not better:
        return "REJECT", f"candidate significantly worse (McNemar p={cmp['p_value']})"
    if significant and better:
        return "PROMOTE", f"candidate significantly better (McNemar p={cmp['p_value']})"
    if not base_ok:
        return "PROMOTE", "baseline fails the gate, candidate passes"
    return "KEEP_BASELINE", f"no significant difference (McNemar p={cmp['p_value']}); keep the smaller model"


def to_markdown(g: dict[str, Any]) -> str:
    lines = [f"# Release gate: {g['decision']}", "", g["reason"], ""]
    for role in ("candidate", "baseline"):
        if not g.get(role):
            continue
        r = g[role]
        lines += [f"## {role}: {r['label']} ({r['model']}, commit {r['git_commit']})", "",
                  "| check | value | limit | ok |", "|---|---|---|---|"]
        lines += [f"| {c['check']} | {c['value']} | {c['op']} {c['limit']} | {'yes' if c['ok'] else 'NO'} |"
                  for c in r["checks"]]
        lines.append("")
    if g.get("comparison"):
        c = g["comparison"]
        lines += ["## Paired comparison (McNemar exact)", "",
                  (f"- paired questions: {c['paired_questions']} (same question set: {c['same_question_set']}, "
                   f"same prompt: {c['same_prompt']})"),
                  f"- only baseline passed: {len(c['only_baseline_passed'])} {c['only_baseline_passed']}",
                  f"- only candidate passed: {len(c['only_candidate_passed'])} {c['only_candidate_passed']}",
                  f"- p-value: {c['p_value']}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--baseline", type=Path)
    args = ap.parse_args()
    t = json.loads(THRESHOLDS.read_text())

    cand = json.loads(args.candidate.read_text())
    cand_checks = threshold_checks(cand, t)
    cand_ok = all(c["ok"] for c in cand_checks)
    base = base_checks = base_ok = cmp = None
    if args.baseline:
        base = json.loads(args.baseline.read_text())
        base_checks = threshold_checks(base, t)
        base_ok = all(c["ok"] for c in base_checks)
        cmp = compare(base, cand)
    decision, reason = decide(cand_ok, base_ok, cmp, t["mcnemar_alpha"])

    def card(r: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
        return {"label": r["label"], "model": r["served"]["model"], "git_commit": r["git_commit"],
                "accuracy": r["accuracy"], "checks": checks}

    gate = {"decision": decision, "reason": reason, "thresholds": t, "candidate": card(cand, cand_checks),
            "baseline": card(base, base_checks) if base else None, "comparison": cmp}
    (REPORTS / "gate.json").write_text(json.dumps(gate, indent=2))
    (REPORTS / "gate.md").write_text(to_markdown(gate))
    print(to_markdown(gate))
    sys.exit(1 if decision in ("REJECT", "FAIL") else 0)


if __name__ == "__main__":
    main()
