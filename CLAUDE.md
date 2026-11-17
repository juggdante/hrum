# CLAUDE.md — HRUM (HR Unified Monitoring & Intelligence)

## What this project is
End-to-end HR data showcase for a Chief HR Officer. Real data: SF Employee Compensation
(data.sfgov.org, `88g8-5mnd`). Synthetic data: HRIS fields (hire/termination, level,
performance, gender, age band). Layers: ingestion → dbt marts → dashboard → ML → LLM agent.
Read `README.md` first for architecture, and `docs/` for details.

## Working language
- Talk to the user in casual Indonesian. Write code, comments, commit messages, and docs in English.
  Exception: `docs/technical_deep_dive.md` (study guide + jury Q&A) is in Indonesian, by the user's choice.

## Environment
- Developed in VS Code on a laptop (no GPU: data, dbt, tests). LLM runs on a local PC with **RTX 4090 24 GB**
  (vLLM); Colab T4 is a fallback for reviewers.
- Warehouse: **DuckDB** file `data/hrum.duckdb` (schemas `raw`, `stg`, `mart`), dbt-duckdb in `warehouse/`.
  BigQuery is an optional extra only. Config in `configs/platform.yaml`.
- LLMs: Qwen3.5-4B (baseline, bf16) and Qwen3.5-9B (candidate, FP8) served by vLLM (OpenAI-compatible),
  plus Anthropic/OpenAI. Routing + vLLM launch profiles in `configs/llm.yaml`.
- This repo is the capstone ("tugas besar") of the Senior AI Engineering Series; course lab functions are
  adapted (rewritten in English) into `.py` modules. Course materials themselves are never committed.
- Python >= 3.11, deps in `pyproject.toml` with exact versions in `uv.lock` (`make setup` = `uv sync --locked`), commands in `Makefile`.

## Hard rules
1. **Never commit secrets.** Use `.env` (gitignored). Only edit `.env.example` for new variables.
2. **Never commit raw data** in `data/raw/` or `data/synthetic/` (small samples go in `data/samples/`).
3. **Layering:** raw is loaded as-is; cleaning only in dbt staging; business logic only in marts/semantic layer.
4. **Single LLM interface:** agent code talks only to `agent/llm/base.py`. No provider SDK calls elsewhere.
5. **Agent safety:** read-only SQL, `hrum_mart` only, enforce limits in `configs/guardrails.yaml`
   (min group size, no individual lookups, bytes-billed cap). Validate SQL with sqlglot before execution.
6. **Synthetic honesty:** every synthetic column/table must be labelled (`is_synthetic`) and documented
   in `docs/data_dictionary.md`. Never describe model output on synthetic fields as real findings.
7. **One definition per metric** in `warehouse/semantic/`. Dashboards, ML features and the agent all reuse it.
8. **Agent answers must be transparent:** show metric definition, query/tool call, caveats.

## Conventions
- dbt naming: `stg_<source>__<table>`, `int_<...>`, `fct_<...>`, `dim_<...>`. Every model has tests + description.
- Python: type hints, `ruff` formatting, small functions, tests in `tests/`.
- Notebooks are for exploration only; production logic lives in `.py` modules.
- Each ingestion source = one module in `ingestion/sources/` with idempotent loads (`_loaded_at`, `_source_file`).
- Prefer simple and explainable over clever (the audience is a CHRO, not an ML conference).

## Key metrics (define in warehouse/semantic/)
total_compensation, avg_total_compensation, benefits_ratio (benefits / salary), overtime_ratio,
headcount, compa_ratio (needs benchmark), turnover_rate (synthetic), cost_of_turnover (synthetic).

## Verify before assuming
- SF Comp header is verified (see `docs/data_dictionary.md`, 26 columns incl. `Employee Name`, `Hours`,
  `Employment Type`, `data_as_of`, `data_loaded_at`). Select columns by name, never by position.
- `Employee Name` is PII: raw/staging only; marts use a hashed `employee_key` or drop it.
- Run the open EDA checks listed in `docs/data_dictionary.md` before designing marts.
- Model names in `configs/llm.yaml` may be outdated. Confirm current names before using.
- The CSV URL in `configs/platform.yaml` should be tested on first ingest.

## Current status / next tasks (update as you go)
Done (2026-11-11): Phase 0 repo hygiene; Phase 1 data foundation — SF Comp ingest, EDA (results in
`docs/data_dictionary.md`), synthetic HRIS, dbt marts (34 tests green), semantic layer + compiler
(`common/semantic.py`, tests in `tests/`).
Done (2026-11-11): Phase 2 code — `infra/local_server/vllm_server.py`, `agent/llm/` (OpenAI-compatible + `get_client()`);
Phase 3 — agent core (`agent/core/loop.py`, tool runner, sqlglot guard, grounding), CLI + Streamlit app.
Done (2026-11-11): Phase 4 RAG (`agent/rag/`, `configs/rag.yaml`, tool `search_docs`, FAQ in `docs/kb/`);
Phase 5 ML attrition (`ml/features/attrition.py`, `ml/attrition/train.py|score.py`, tool `attrition_insight`).
Done (2026-11-11): Phase 6 MCP gateway (`agent/mcp/gateway.py|server.py`, pins in `agent/mcp/tool_pins.json`;
controls listed in `docs/governance.md`). Verified over Streamable HTTP with valid/invalid JWTs.
65 pytest tests green using a scripted fake LLM.
Done (2026-11-11): first real-model run on the 4090 (Qwen3.5-4B bf16, vLLM 0.30.0): bench TTFT p50 0.026 s,
~102 decode tok/s; `agent.cli --smoke` answered with grounding ok, repair 0, 3 LLM calls, 3.9 s.
4090 PC quirk: system nvcc is too old for FlashInfer JIT, so the launcher sets VLLM_USE_FLASHINFER_SAMPLER=0.
Main branch `main` tracks `origin` (office GitLab). Pull it on the 4090 PC.

Done (2026-11-11): synthetic generator + ML training made deterministic (fixed row order); retrained:
XGBoost ROC-AUC 0.722, lift 2.85. `attrition_insight` default 'all' = one-call overview.
Done (2026-11-11): Phase 7 harness — `evals/golden_questions.yaml` (39 items, gold from SQL), `evals/checks.py`,
`stats.py` (Wilson, exact McNemar), `run_eval.py`, `gate.py`, `thresholds.json`. Verified with a fake LLM only.

Done (2026-11-12): Phase 7. Final eval (run 3, prompt v3, regraded): 4B and 9B both 39/39, gate KEEP_BASELINE
(4B ships). Reports committed in `evals/reports/`. Details in `docs/progress_report.md`.

Next (plan: `~/.claude/plans/saya-mau-kerjain-tugas-besar-txt-indexed-naur.md`):
Done (2026-11-12): decimal-comma normalisation (`agent/guardrails/number_style.py`); Phase 8 code — Langfuse
tracing + PII masking (`agent/observability/`, `infra/langfuse/README.md`, `make trace-check`), 112 tests.
Done (2026-11-12): Phase 8 verified on the 4090 — self-hosted Langfuse (Docker, ~/langfuse, ports on 127.0.0.1),
traces + masking checked in the UI. Retrieval warm-up in app and eval.
Done (2026-11-12): app v2 — multipage Streamlit (`agent/app/main.py` + `views/`), Plotly charts from cited
evidence (`agent/app/charts.py`), SQL Explorer with `check_sql(policy="explore")`; 147 tests.
Done (2026-11-12): app v2 clicked through on the 4090; `docs/setup_guide.md` + `make doctor` (`infra/doctor.py`).
Done (2026-11-13..16): deck (`docs/deck/`), `docs/technical_deep_dive.md` (ID), `docs/demo_runbook.md` +
`make demo-check`, roadmap in README, RAG sources pinned to the evaluated set, `docs/architecture.md` rewritten.
Done (2026-11-17): capstone submitted — repo published on GitHub, deck exports in `docs/deck/`, LinkedIn post.
Also: `uv.lock` (CPU torch, Python >= 3.11), tests never export to Langfuse, doctor finds HRUM's own app port.
Next: 1. Fresh-clone test of the setup guide on the 4090 (in progress, user reports back). 2. Branching plan
(tag checkpoint, feature/multi-turn): `private/hrum_branching_plan.md`, when the user is ready.
3. Roadmap (README), each eval-gated. 4. Prompts: caveat consistency.

Keep `docs/progress_report.md` (milestones, results) and `docs/lessons_learned.md` (dos/don'ts/gotchas) current.

## Commands
`make setup | lock-upgrade | build-data | index | ingest | synth | dbt-run | dbt-test | test | train | serve | agent | mcp | eval | regrade | failures | gate | trace-check | doctor | demo-check`
