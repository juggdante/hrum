"""Print the failed questions of the latest eval run per model, with answers, tools and grounding problems.

    python -m evals.show                      # all models with runs
    python -m evals.show qwen3.5-4b --all     # every question, not only failures
"""
from __future__ import annotations

import argparse
import json

from evals.run_eval import latest_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for label, f in latest_runs(args.labels).items():
        print(f"\n######## {label}  ({f.name})")
        for line in f.open():
            r = json.loads(line)
            g, res = r["grade"], r["result"]
            if g["passed"] and not args.all:
                continue
            print(f"\n== {g['id']} [{res['status']}] {'PASS' if g['passed'] else 'FAIL'} {g['problems']}")
            print("   Q:", r["question"])
            print("   A:", res.get("answer", "")[:400])
            for e in res.get("evidence", []):
                print("  ", e["id"], e["tool"], json.dumps(e["arguments"], ensure_ascii=False)[:160])
            for p in g.get("grounding_problems", []):
                print("   grounding:", p)
            if res.get("error"):
                print("   error:", res["error"])


if __name__ == "__main__":
    main()
