"""Run the golden questions against one model and write a report (adapted from course lab P10 `ci_eval.py`).

    python -m evals.run_eval                          # configured provider, label from the served model
    python -m evals.run_eval --only cost_01,atk_01    # a subset
    python -m evals.run_eval --repeat 3               # pass^k: a question passes only if all k runs pass

Writes evals/reports/runs/<label>-<timestamp>.jsonl (every answer, gitignored) and
evals/reports/<label>.json (summary + per-question pass/fail, used by evals.gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import requests

from agent.core.loop import answer_question
from agent.llm.base import LLMClient, get_client
from agent.observability import tracing
from agent.prompts import PROMPT_VERSION
from common.config import ROOT, db_path, load_config
from evals.checks import QUESTIONS, Gold, compute_gold, grade, load_questions
from evals.stats import percentile, wilson

REPORTS = ROOT / "evals" / "reports"


def served_model(provider: str) -> dict[str, Any]:
    """Ask the OpenAI-compatible server which weights are loaded (vLLM reports them as `root`)."""
    p = load_config("llm")["providers"][provider]
    if not p.get("base_url"):
        return {"model": p.get("model")}
    try:
        r = requests.get(f"{p['base_url'].rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {p.get('api_key', '')}"}, timeout=5)
        m = r.json()["data"][0]
        return {"model": m.get("root") or m.get("id"), "max_model_len": m.get("max_model_len")}
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return {"model": p.get("model")}


def preflight_error(base_url: str | None, api_key: str = "") -> str | None:
    """Why the model server cannot serve an eval, or None when it can. Cloud providers (no base_url) are skipped.

    Without this check an unreachable server turns every question into an `error` and the report shows 0/N.
    """
    if not base_url:
        return None
    url = f"{base_url.rstrip('/')}/models"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
    except requests.RequestException:
        return f"model server not reachable at {base_url}: start it with `make serve` (status: " \
               "`.venv/bin/python -m infra.local_server.vllm_server status`)"
    if r.status_code == 401:
        return "the model server rejected QWEN_API_KEY: .env does not match the key the server was started with"
    if r.status_code != 200:
        return f"model server answered HTTP {r.status_code} at {url}"
    return None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.split("/")[-1].lower()).strip("-")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_one(item: dict[str, Any], llm: LLMClient, run_tag: str = "") -> dict[str, Any]:
    t0 = time.time()
    trace = {"session_id": run_tag or None, "tags": ["eval", item["category"]],
             "metadata": {"question_id": item["id"], "category": item["category"]}}
    try:
        return answer_question(item["question"], llm, role=item.get("role", "chro"), trace=trace).to_dict()
    except Exception as e:  # noqa: BLE001 - an eval must record crashes, not stop on them
        return {"status": "error", "answer": "", "claims": [], "evidence": [], "grounding": {"ok": False},
                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=5),
                "usage": {"seconds": round(time.time() - t0, 2)}}


def grade_record(item: dict[str, Any], result: dict[str, Any], gold: Gold, rep: int = 0) -> dict[str, Any]:
    g = grade(item, result, gold)
    u = result.get("usage", {})
    g.update(repeat=rep, seconds=u.get("seconds", 0), llm_calls=u.get("llm_calls", 0),
             tokens=u.get("prompt_tokens", 0) + u.get("completion_tokens", 0),
             repair_level=result.get("repair_level", 0),
             grounding_ok=bool(result.get("grounding", {}).get("ok")),
             grounding_problems=result.get("grounding", {}).get("problems", []),
             llm_seconds=round(u.get("llm_seconds", 0), 3),
             tool_seconds=round(sum(ev.get("seconds", 0) for ev in result.get("trace", [])
                                    if ev.get("kind") == "tool"), 3))
    return g


def latest_runs(labels: list[str] | None = None) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for f in sorted((REPORTS / "runs").glob("*.jsonl")):
        label = f.stem.rsplit("-", 1)[0]
        if not labels or label in labels:
            runs[label] = f  # file names sort by timestamp, so the last one wins
    return runs


def regrade(label: str) -> dict[str, Any]:
    """Re-grade the latest saved run of `label` with the current checks. No model calls, no GPU.

    Use after fixing a grader bug. Answers and gold values come from the run file; question specs
    (expected status, must_include, ...) from the current golden_questions.yaml.
    """
    run_file = latest_runs([label]).get(label)
    report = REPORTS / f"{label}.json"
    if run_file is None or not report.exists():
        raise SystemExit(f"no saved run/report for '{label}'")
    items = {q["id"]: q for q in load_questions()}
    graded, rep_count = [], defaultdict(int)
    for line in run_file.open():
        rec = json.loads(line)
        qid = rec["grade"]["id"]
        graded.append(grade_record(items[qid], rec["result"], Gold(**rec["gold"]), rep_count[qid]))
        rep_count[qid] += 1
    old = json.loads(report.read_text())
    meta = {k: old[k] for k in ("label", "provider", "served", "git_commit", "prompt_version", "timestamp",
                                "repeat", "subset", "run_file") if k in old}
    meta.update(questions_sha=hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()[:12],
                regraded_with=git_commit())
    summary = summarize(graded, meta)
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(graded: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in graded:
        by_q[g["id"]].append(g)
    per_question = {qid: all(r["passed"] for r in runs) for qid, runs in by_q.items()}
    first = [runs[0] for runs in by_q.values()]

    cats: dict[str, list[bool]] = defaultdict(list)
    for r in first:
        cats[r["category"]].append(per_question[r["id"]])
    attacks = [r for r in graded if r["attack_breach"] is not None]
    answered = [r for r in graded if r["status"] == "answered"]
    tool_checks = [r["tool_selection_ok"] for r in graded if r["tool_selection_ok"] is not None]
    secs = [r["seconds"] for r in graded]
    style_keys = ["decimal_comma", "no_thousands_separator", "glued_sentence"]

    return {
        **meta,
        "accuracy": _rate(sum(per_question.values()), len(per_question)),
        "by_category": {c: _rate(sum(v), len(v)) for c, v in sorted(cats.items())},
        "attack_breach": _rate(sum(r["attack_breach"] for r in attacks), len(attacks)),
        "grounding_ok_when_answered": _rate(sum(r["grounding_ok"] for r in answered), len(answered)),
        "tool_selection": _rate(sum(tool_checks), len(tool_checks)),
        "style_issue_rate": {k: _rate(sum(r["style"].get(k, False) for r in answered), len(answered))["rate"]
                             for k in style_keys},
        "errors": sum(r["status"] == "error" for r in graded),
        "latency_s": {"p50": round(percentile(secs, 50), 2), "p95": round(percentile(secs, 95), 2),
                      "max": round(max(secs, default=0), 2)},
        "stage_seconds_mean": {
            "llm": round(sum(r.get("llm_seconds", 0) for r in graded) / max(len(graded), 1), 2),
            "tools": round(sum(r.get("tool_seconds", 0) for r in graded) / max(len(graded), 1), 2),
            "other": round(sum(max(r["seconds"] - r.get("llm_seconds", 0) - r.get("tool_seconds", 0), 0)
                               for r in graded) / max(len(graded), 1), 2)},
        "avg_llm_calls": round(sum(r["llm_calls"] for r in graded) / max(len(graded), 1), 2),
        "avg_tokens": round(sum(r["tokens"] for r in graded) / max(len(graded), 1)),
        "repairs": sum(r["repair_level"] > 0 for r in graded),
        "per_question": per_question,
        "failures": {r["id"]: r["problems"] for r in graded if not r["passed"]},
    }


def print_summary(s: dict[str, Any]) -> None:
    a = s["accuracy"]
    print(f"\n{s['label']}  ({s['served']['model']}, commit {s['git_commit']}, prompt {s['prompt_version']})")
    print(f"accuracy        {a['k']}/{a['n']} = {a['rate']:.1%}  95% CI {a['ci95'][0]:.0%}-{a['ci95'][1]:.0%}")
    for c, r in s["by_category"].items():
        print(f"  {c:<12}  {r['k']}/{r['n']}")
    b = s["attack_breach"]
    print(f"attack breach   {b['k']}/{b['n']}")
    g = s["grounding_ok_when_answered"]
    print(f"grounding ok    {g['k']}/{g['n']}   tool selection {s['tool_selection']['k']}/{s['tool_selection']['n']}")
    print(f"latency         p50 {s['latency_s']['p50']}s  p95 {s['latency_s']['p95']}s   "
          f"avg tokens {s['avg_tokens']}  repairs {s['repairs']}  errors {s['errors']}")
    st = s.get("stage_seconds_mean", {})
    print(f"time per answer llm {st.get('llm')}s  tools {st.get('tools')}s  other {st.get('other')}s (mean)")
    print(f"style issues    {s['style_issue_rate']}")
    for qid, problems in s["failures"].items():
        print(f"  FAIL {qid}: {'; '.join(problems)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=None, help="provider in configs/llm.yaml (default: its `default`)")
    ap.add_argument("--label", default=None, help="report name (default: served model, e.g. qwen3.5-4b)")
    ap.add_argument("--only", default=None, help="comma-separated question ids")
    ap.add_argument("--category", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--regrade", nargs="+", metavar="LABEL",
                    help="re-grade the latest saved run(s) with the current checks, without calling the model")
    args = ap.parse_args()
    if args.regrade:
        for label in args.regrade:
            print_summary(regrade(label))
        return

    provider = args.provider or load_config("llm")["default"]
    p = load_config("llm")["providers"][provider]
    problem = preflight_error(p.get("base_url"), p.get("api_key", ""))
    if problem:
        raise SystemExit(f"Not running the eval: {problem}")
    served = served_model(provider)
    label = args.label or slug(served["model"] or provider)
    items = load_questions()
    if args.only:
        wanted = set(args.only.split(","))
        items = [q for q in items if q["id"] in wanted]
    if args.category:
        items = [q for q in items if q["category"] == args.category]
    if not items:
        raise SystemExit("no questions selected")

    llm = get_client(provider)
    from agent.rag.retrieve import warm_up

    warm_up()   # load the embedding model before timing starts, so latency reflects answering, not loading
    con = duckdb.connect(str(db_path()), read_only=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (REPORTS / "runs").mkdir(parents=True, exist_ok=True)
    run_file = REPORTS / "runs" / f"{label}-{stamp}.jsonl"

    graded = []
    with run_file.open("w") as out:
        for i, item in enumerate(items, 1):
            gold = compute_gold(con, item)
            for rep in range(args.repeat):
                result = run_one(item, llm, run_tag=f"eval-{label}-{stamp}")
                g = grade_record(item, result, gold, rep)
                graded.append(g)
                out.write(json.dumps({"grade": g, "question": item["question"], "gold": gold.__dict__,
                                      "result": result},   # full trace kept: runs/ is local only
                                     default=str, ensure_ascii=False) + "\n")
                mark = "PASS" if g["passed"] else "FAIL"
                print(f"[{i:>2}/{len(items)}] {mark} {item['id']:<11} {g['status']:<8} {g['seconds']:>6.1f}s "
                      f"{'; '.join(g['problems'])[:110]}", flush=True)
    con.close()
    tracing.flush()

    meta = {"label": label, "provider": provider, "served": served, "git_commit": git_commit(),
            "prompt_version": PROMPT_VERSION,
            "timestamp": stamp, "repeat": args.repeat, "subset": bool(args.only or args.category),
            "questions_sha": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()[:12],
            "run_file": str(run_file.relative_to(ROOT))}
    summary = summarize(graded, meta)
    report = REPORTS / (f"{label}.json" if not summary["subset"] else f"{label}-subset.json")
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print_summary(summary)
    print(f"\nreport: {report.relative_to(ROOT)}   answers: {run_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
