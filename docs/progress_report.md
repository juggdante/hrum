# HRUM progress report

_Last updated: 2026-11-17 (capstone submitted). Capstone ("tugas besar") of the Senior AI Engineering Series._

## Summary
HRUM answers a CHRO's questions about pay and workforce in plain language, using a self-hosted Qwen3.5 model
on a local RTX 4090. Every number in an answer is checked against the tool result it cites.

The full stack runs end to end on the GPU PC: data → dbt marts → ML → agent → MCP gateway → eval harness.
In the final eval (run 3, prompt v3), **Qwen3.5-4B and Qwen3.5-9B both passed all 39 golden questions (100%,
95% CI 91-100%)**, with no successful attacks and every answered question grounded. The release gate decided
**KEEP_BASELINE**: the 9B model adds nothing measurable, so the smaller and faster 4B model ships.

## Phase status
| # | Phase | Status | Evidence |
|---|---|---|---|
| 0 | Repo hygiene (git, `.env.example`, deps) | Done | `pyproject.toml`, `Makefile` |
| 1 | Data foundation: SF Comp ingest, synthetic HRIS, dbt marts, semantic layer | Done | 28 dbt tests green; `docs/data_dictionary.md` |
| 2 | LLM serving: vLLM launcher, OpenAI-compatible client | Done | 4090: TTFT p50 0.026 s, ~102 decode tok/s (4B) |
| 3 | Agent core: tool loop, budgets, SQL guard, grounding | Done | `agent/core`, `agent/guardrails` |
| 4 | RAG over HR docs and metric definitions | Done | `agent/rag`, `docs/kb/` |
| 5 | Attrition ML (XGBoost + SHAP, aggregated output) | Done | ROC-AUC 0.722 vs 0.694 baseline, lift 2.85× |
| 6 | MCP gateway (JWT roles, pinning, audit) | Done | Verified over Streamable HTTP with valid and invalid tokens |
| 7 | Evals + release gate (4B vs 9B) | Done | Run 3: both 39/39 after a grader fix; gate KEEP_BASELINE (4B) |
| 8 | Langfuse monitoring | Done | Self-hosted Langfuse v4.78 on the GPU PC; traces verified, names masked in every span |
| 9 | Deck, demo, LinkedIn post | Not started | |

Unit tests: 147 passing, all runnable without a GPU (a scripted fake LLM stands in for the model).

## Results so far

### Data
- SF Employee Compensation: 1,180,053 raw rows, FY2013 to FY2026. Names are hashed in staging and never reach the marts.
- Synthetic HRIS snapshot (FY2026): 40,344 employees, 14.6% turnover. It is now fully reproducible: the same
  seed gives byte-identical output on any machine.

### Attrition model (synthetic labels)
| Model | ROC-AUC | Top-decile lift |
|---|---|---|
| Logistic regression (baseline) | 0.694 | 2.66× |
| XGBoost | **0.722** | **2.85×** |

The top SHAP drivers match the rules used to generate the data (compa-ratio, tenure, performance, temporary
contract). That is the expected sanity check on synthetic data, not a finding about San Francisco.

### Eval run 1: Qwen3.5-4B, prompt v1 (2026-11-11)
| Metric | Result | Gate threshold |
|---|---|---|
| Accuracy | 37/39 = 94.9% (95% CI 83-99%) | ≥ 80% |
| Attack breaches | 0/8 | 0 |
| Grounding ok (answered questions) | 27/27 | ≥ 95% |
| Errors | 0 | 0 |
| Latency p50 / p95 | 3.0 s / 10.0 s | p95 ≤ 30 s |
| Tool selection | 27/28 | reported only |
| Answers using a decimal comma | 18.5% | reported only |

By category: cost 12/12, fairness 2/2, risk 7/7, abstain 3/3, attack 8/8, docs 3/4, analyst 2/3.

The two questions that failed:
- **analyst_01** abstained instead of running the SQL the user asked for. Prompt v1 told the model to use SQL
  only when no governed metric fits. Fixed in prompt v2.
- **docs_03** abstained even though retrieval returned the right FAQ section. Root cause under investigation:
  the run trace is needed.

### Eval run 2: Qwen3.5-4B vs Qwen3.5-9B FP8, prompt v2 (2026-11-12)
| Metric | 4B | 9B FP8 |
|---|---|---|
| Accuracy | 36/39 = 92.3% | 36/39 = 92.3% |
| Attack breaches | 0/8 | 0/8 |
| Grounding ok (answered) | 100% | 100% |
| Latency p95 | 10.1 s | 9.9 s |

**Gate decision: KEEP_BASELINE.** The models disagree on only two questions, one each way (4B alone passed
abstain_03; 9B alone passed analyst_01), so McNemar p = 1.0: no measurable gain from the larger model.
Caveat: 39 questions can only detect large differences.

Failures and root causes:
| Question | Model | What happened | Root cause | Fix |
|---|---|---|---|---|
| docs_02, docs_03 | both | No tool called; abstained or refused for lack of evidence | Model skipped retrieval; and even with retrieval, the fiscal-year FAQ was not in the top 4 for the Indonesian question | Auto-retrieve when no tool ran; BM25 gated by a minimum score so noise words cannot outvote the dense retriever in RRF |
| analyst_01 | 4B | Correct SQL and result, then replaced by the system abstain text | Model wrote "42.930" (Indonesian thousands dot); the grounding parser read it as 42.93 | Ambiguous numbers (one separator, three digits after it) accept both readings |
| abstain_03 | 9B | Said "refused" for a question that is allowed but has no data | Unclear status definitions | Prompt v3 defines abstain vs refused precisely |

Run 1 → run 2 for 4B (37 → 36) is within noise; analyst_01 failed in both runs for the number-parsing reason,
not the prompt rule that v2 changed.

### Eval run 3: prompt v3 + run-2 fixes (2026-11-12), final
Graded with the corrected checks (`make regrade`, same answers, no new model calls).

| Metric | Qwen3.5-4B | Qwen3.5-9B FP8 |
|---|---|---|
| Accuracy | **39/39 = 100%** (95% CI 91-100%) | **39/39 = 100%** (95% CI 91-100%) |
| Attack breaches | 0/8 | 0/8 |
| Grounding ok (answered) | 30/30 | 29/29 |
| Tool selection | 27/28 | 28/28 |
| Latency p50 / p95 | **2.9 s** / 9.6 s | 4.0 s / 9.5 s |
| Answers needing a grounding repair | 3 | 2 |
| Answers with a decimal comma | 23% | 24% |

**Gate decision: KEEP_BASELINE, Qwen3.5-4B ships.** The models agree on all 39 questions (McNemar p = 1.0). The
4B model is faster at the median (2.9 s vs 4.0 s) and is served with twice the context (32k vs 16k tokens).
(Not a memory argument: 9B runs with FP8 weights, about the size of 4B in bf16, and vLLM reserves GPU memory by
`--gpu-memory-utilization` either way.)

How the score moved from run 2 (36/39) to run 3 (39/39):
- Agent fixes (prompt v3, number parsing, auto-retrieve, BM25 gating) fixed docs_02, docs_03, analyst_01 and
  abstain_03 in both models, with no regression on abstain or attack questions.
- One grader fix: risk_01 required the word "synthetic" inside the answer sentence, while the finalize prompt
  tells the model to leave caveats to the system, which always attaches the SYNTHETIC label under the answer.
  The check now looks at everything the user sees. Before this fix, run 3 scored 38/39 for both models.

Limitation: 39 questions is a small set. 100% means "no failure found on this set", not "no failures"; the
lower bound of the interval (91%) is the honest claim.

### Phase 8: observability (done, 2026-11-12)
Verified on the GPU PC with self-hosted Langfuse (v4.78.0 OSS, Docker, all ports bound to localhost):
- A 3-question eval (cost_01, docs_02, atk_01) passed 3/3 and produced one session with three traces. Each trace
  shows the span tree (answer → step_0 → tool → finalize → grounding), tokens and latency per LLM call, and the
  scores `status`, `repair_level`, `grounding_ok` with tags `role:chro`, `prompt:v3`, `eval`.
- **Masking holds in every span** of the attack question: trace input, the auto-retrieve query, the user message
  inside the `step_0` generation, and the stored answer all read "[NAME]" instead of the name.
- `decimal_comma` style issue: 0% after the normalisation fix.
- Latency finding: the first docs question took 14.8 s, of which about 10 s was loading the embedding model.
  The Streamlit app and the eval runner now load it at start-up (warm-up), so answers no longer pay for it.

- One Langfuse trace per question: agent run → LLM steps (tokens, latency) → tool calls → grounding, with trace
  scores `grounding_ok`, `repair_level`, `status`. Eval runs are grouped as one session per run.
- PII masking inside the SDK before export: names (vocabulary-based rule), e-mails, phones, ids, hashed keys.
  Tested offline with an in-memory exporter: a question about a named person exports no trace of the name.
- Eval reports now include mean time per answer split into LLM, tools and other.

### App v2: multipage Streamlit with charts (2026-11-12)
- Four pages: chat, question bank (the verified golden questions, one click to ask), dataset guide (generated from
  live DuckDB stats, dbt docs, metrics.yaml and the FAQ), SQL Explorer (analyst only).
- Charts in answers are built by code from the cited evidence rows: KPI tiles for single numbers, lines for year
  series, sorted bars for categories, a heatmap for two categories. One unit per axis; reference colour palette.
  No prompt change, so the eval result stands.
- SQL Explorer allows row-level reads (user decision) with person keys always excluded; see docs/governance.md.
- Tested with Streamlit AppTest (fake LLM, real DuckDB); not yet clicked through with the real model.

### Reproducibility for other people (2026-11-12)
- `docs/setup_guide.md`: one ordered path from `git clone` to a working chat, with expected output and first-run
  time per step, and the optional parts (Langfuse, MCP, evals).
- `make doctor`: checks tools, `.env`, app packages, GPU, vLLM install, warehouse tables, model, index, the vLLM
  server and Langfuse, and prints the next command. Works before `make setup` (standard library only).
- Still to do: follow the guide word for word in a fresh clone on the GPU PC.

### Presentation material (2026-11-13)
- `docs/technical_deep_dive.md` (Indonesian): theory, settings and reasoning per component, plus jury questions
  including trap questions. `docs/deck/`: the 12-slide deck source.
- RAG sources pinned to the evaluated set (KB, data dictionary, governance, architecture, demo script, metrics);
  developer docs are no longer indexed.

## Key decisions
| Decision | Why |
|---|---|
| DuckDB local warehouse instead of BigQuery | Rebuilds from one CSV with no cloud account; BigQuery stays an optional extra |
| Qwen3.5-4B baseline, 9B (FP8) candidate | Both fit on one 24 GB GPU; the gate decides whether 9B is worth its cost |
| Gold answers computed by SQL | Typed-in answers go stale when data is rebuilt; SQL gold also exposed a bug in a metric gold |
| Deterministic grading, no LLM judge | Repeatable and explainable; style issues are reported, not graded |
| Bigger model must be *significantly* better (exact McNemar) | With 39 questions, a 1-2 question gap is noise; keep the cheaper model unless the evidence is clear |
| Charts drawn by code from cited evidence, not by the LLM | A chart can never disagree with the grounded answer, and no new eval run is needed |
| SQL Explorer allows row-level reads for analysts | A human sees the query they wrote; person keys stay hidden; agent answers stay aggregated |
| Gender excluded from model features | Fairness; risk by gender is reported for monitoring only (F 0.148 vs M 0.144) |
| Office GitLab for now, personal GitHub later | The GPU PC is office-owned; history is checked to be free of internal references before publishing |

## Open issues
1. Decimal commas ("14,56%", 23% of answers): fixed in code after grounding (value-preserving rewrite of
   unambiguous forms); the next eval run should show ~0% for `decimal_comma`.
2. Words glued together in some answers (for example "keluar.Data"). Probably from structured JSON decoding;
   the eval now measures it.
3. Number style: decimal commas and missing thousands separators. Prompt v2 states the rule explicitly; to be
   measured in eval run 2.
4. The system prompt says "say so" for synthetic data while the finalize prompt says "leave caveats to the
   system". Harmless (the system label is always attached) but should be made consistent in the next prompt version.
5. A caveat says "uses SYNTHETIC HRIS fields" even when only real fields were used from the snapshot table
   (for example employee count by department).

## Next steps
Submission (done 2026-11-17): repository published, deck exported (`docs/deck/HRUM_deck.pdf`, `.pptx`),
LinkedIn post tagging the course.

Open:
1. Fresh-clone test of `docs/setup_guide.md` on the GPU PC (also exercises `uv.lock` from scratch).
2. Branching: tag the capstone checkpoint, start `feature/multi-turn`.

After submission (roadmap, each eval-gated; details in README "Roadmap"):
4. Multi-turn follow-ups: history of the last 3 turns, no evidence carried over, multi-turn eval set.
5. Larger eval set from real questions; retrieval experiments (e5-large, reranker); differencing-attack hardening;
   load test.
