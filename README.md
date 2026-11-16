# HRUM — HR Unified Monitoring & Intelligence

"Talk to your HR data" for a Chief HR Officer, answered by an **on-premise LLM** (Qwen3.5 on vLLM).
Every number in an answer is grounded in a cited tool result, and access is controlled by an MCP gateway.

> **Data honesty:** compensation & benefits data is **real** (City & County of San Francisco,
> public). HRIS fields such as hire/termination date, performance, level and gender are **synthetic**.
> The attrition model and turnover metrics on synthetic fields demonstrate capability, not findings.

## CHRO storyline
1. **Cost**: How much do we pay, to whom, and what is the trend? (salary vs benefits vs overtime, FY2013-2026)
2. **Fairness**: Is pay consistent within a job? (internal compa-ratio)
3. **Risk**: Which segments are likely to leave, why, and what does it cost? (attrition model + SHAP, synthetic)
4. **Ask**: "Tanya aja ke datanya". Natural-language Q&A with transparent, auditable answers.

## Architecture
```
SF Employee Compensation (real) ─┐
Synthetic HRIS layer ────────────┴─► ingestion ─► DuckDB raw ─► dbt staging ─► marts + semantic layer
                                                                                     │
HR docs + metric definitions ─► RAG index (BM25 + e5, RRF) ──┐                       │
Attrition model (XGBoost + SHAP, aggregated) ────────────────┤                       │
                                                             ▼                       ▼
         vLLM: Qwen3.5-4B / 9B  ◄──  agent loop (tool calling, budgets, grounding)  ── tools
                                                │
                       MCP gateway (JWT role scope, allowlist, pinning, injection scan, audit)
                                                │
                        Streamlit chat · CLI · any MCP client
```

## Stack
- **Warehouse:** DuckDB file `data/hrum.duckdb` (schemas `raw`, `stg`, `mart`); BigQuery is an optional extra.
- **Transform:** dbt-duckdb with a YAML semantic layer (`warehouse/semantic/metrics.yaml`, one definition per metric).
- **ML:** XGBoost vs logistic regression, SHAP, MLflow (SQLite backend).
- **LLM:** Qwen3.5-4B (baseline, bf16) and Qwen3.5-9B (candidate, FP8) on vLLM, OpenAI-compatible.
  Anthropic/OpenAI can be swapped in through `configs/llm.yaml`.
- **Agent:** tool calling with step/tool/time budgets, sqlglot SQL guard, numeric grounding, RAG, MCP (`mcp` SDK).
- **Runtime:** a local RTX 4090 24 GB PC for the model; data, dbt and tests run on any CPU machine.

## What the agent enforces
| Control | Where |
|---|---|
| Numbers come only from tools; every claim cites evidence (E1, E2, ...), unsupported claims are repaired or dropped | `agent/guardrails/grounding.py`, `agent/core/loop.py` |
| Read-only SQL, `mart` schema only, no PII columns, aggregates only, groups of at least 5, row limit | `agent/guardrails/sql_guard.py`, `configs/guardrails.yaml` |
| Roles: CHRO uses the semantic layer only; analyst may also run guarded SQL | `configs/guardrails.yaml`, `agent/tools/` |
| Retrieved documents are sanitised; tool output is treated as data, not instructions | `agent/rag/retrieve.py`, `agent/prompts/` |
| MCP: JWT audience/issuer/expiry, scope → role, tool fingerprint pinning, rate limit, JSONL audit | `agent/mcp/`, `docs/governance.md` |
| Traces (self-hosted Langfuse) with names, e-mails, phones and hashed keys masked before export | `agent/observability/` |

## Repo layout
| Path | Purpose |
|---|---|
| `configs/` | Platform, LLM providers + vLLM profiles, guardrails, RAG (YAML) |
| `common/` | Config loader, semantic-layer compiler |
| `ingestion/` | SF Comp loader, synthetic HRIS generator |
| `warehouse/` | dbt project: staging, marts, tests, semantic layer |
| `ml/` | Attrition features, training, aggregated scoring; model card in `ml/registry/` |
| `agent/` | LLM client, agent loop, tools, guardrails, RAG, MCP server, CLI, Streamlit app |
| `evals/` | Golden questions with SQL-computed gold answers, eval runner, release gate |
| `infra/local_server/` | vLLM launcher for the 4090 PC ([setup guide](infra/local_server/README.md)) |
| `infra/langfuse/` | Self-hosted Langfuse for traces ([setup guide](infra/langfuse/README.md)) |
| `docs/` | [Setup guide](docs/setup_guide.md), [demo runbook](docs/demo_runbook.md), [technical deep dive (ID)](docs/technical_deep_dive.md), architecture, data dictionary, governance, HR knowledge base, demo script, [progress report](docs/progress_report.md), [lessons learned](docs/lessons_learned.md) |
| `tests/` | pytest suite (runs without a GPU, using a scripted fake LLM) |

## Quick start
**Full walkthrough from `git clone` to a working chat: [docs/setup_guide.md](docs/setup_guide.md).** Running a live demo (terminals, ports, `make demo-check`, fallbacks): [docs/demo_runbook.md](docs/demo_runbook.md). `make doctor` checks every step and prints the next command.

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/). The data steps run on CPU.

```bash
cp .env.example .env            # set QWEN_API_KEY (any random string); HRUM_JWT_SECRET for MCP
make setup                      # .venv with app deps
make build-data                 # download SF Comp (~400 MB) -> raw -> staging -> synthetic -> marts + dbt tests
                                # set SF_COMP_CSV=/path/to.csv in .env to reuse a downloaded file
make train                      # attrition model -> mart.ml_attrition_* (aggregated)
make index                      # RAG index over docs/ and metric definitions
make test                       # pytest, no GPU needed
```

Model serving (GPU PC, vLLM in its own venv; details in [infra/local_server/README.md](infra/local_server/README.md)):
```bash
# once: vLLM pins its own torch, so it gets a separate venv
uv venv ~/.venvs/vllm --python 3.12 && uv pip install --python ~/.venvs/vllm/bin/python "vllm==0.30.0"
# ~/.venvs/vllm is found automatically (elsewhere: VLLM_BIN in .env). Use tmux for long runs over SSH.

make serve                          # Qwen3.5-4B (default profile); progress: tail -f logs/vllm.log
make serve PROFILE=qwen35_9b_fp8    # Qwen3.5-9B FP8 candidate
.venv/bin/python -m agent.cli --smoke
.venv/bin/python -m agent.cli "Berapa total kompensasi FY2025 per organization group?" --role chro
make agent                          # Streamlit app on 127.0.0.1:8501 (APP_PORT=... if taken)
make mcp                            # MCP server over Streamable HTTP (JWT); TRANSPORT=stdio for local clients
```

Step by step, `make build-data` = `make ingest` → `make synth` (builds SF Comp staging, then the synthetic layer) →
`dbt build`. If you run the steps by hand, use `make ingest && make synth && make dbt-run && make dbt-test`.

## The app (`make agent`)
| Page | What it does |
|---|---|
| Tanya HRUM | Chat. Each answer shows status, a Plotly chart or KPI tiles built **from the cited evidence rows** (never drawn by the LLM), caveats, and the evidence: metric definitions, SQL, rows, grounding |
| Daftar pertanyaan | The 39 verified golden questions by topic, with eval status; one click sends a question to the chat (switching role when needed) |
| Panduan dataset | Real vs synthetic, live coverage numbers, metric catalogue, mart tables and columns, dimension values, FAQ |
| SQL Explorer | Analyst role only: read-only SQL on `mart`, row-level allowed, person keys always hidden, chart + CSV download |

## Evaluation and release gate
`evals/golden_questions.yaml` holds 39 questions across cost, fairness, risk, docs, analyst SQL, abstain and
attack categories. Gold answers are computed from the warehouse with `gold_sql` (never typed in), so they stay
correct after a data rebuild. Grading is deterministic: numbers within 1%, names as text, required phrases,
grounding, plus leak checks (no person-level numbers, no hashed keys).

```bash
make serve                          # baseline on the GPU
make eval                           # -> evals/reports/qwen3.5-4b.json
make serve PROFILE=qwen35_9b_fp8
make eval                           # -> evals/reports/qwen3.5-9b.json
make gate                           # thresholds + paired McNemar test -> evals/reports/gate.md
```

The gate (`evals/thresholds.json`) requires accuracy >= 80%, zero attack breaches, grounding >= 95% of answers,
no crashes and p95 latency <= 30 s. The bigger model is promoted only if it is significantly better on the same
questions; otherwise the smaller, cheaper model stays.

## Status
- [x] Data foundation: ingest, synthetic HRIS, dbt marts + tests, semantic layer
- [x] LLM serving + agent core: vLLM launcher, OpenAI-compatible client, tool loop, SQL guard, grounding
- [x] RAG over HR docs and metric definitions
- [x] Attrition ML (XGBoost ROC-AUC 0.72 on the synthetic holdout), aggregated risk segments + drivers
- [x] MCP gateway with role-scoped JWT and audit
- [x] First run against the real model on the 4090
- [x] Eval harness + release gate (39 questions, McNemar)
- [x] Eval results: Qwen3.5-4B and 9B both 39/39; release gate keeps 4B
- [x] Langfuse tracing with PII masking (self-hosted on the GPU PC)
- [x] Streamlit app (chat with evidence charts, question bank, dataset guide, SQL Explorer), deck
- [ ] Repository published, LinkedIn post

## Roadmap
Each item ships only after it passes the eval release gate (`make eval`, `make gate`).
1. **Multi-turn follow-ups** ("and per organization group?", "show that for FY2025"). Today every question is
   answered on its own. Design: send the last 3 turns (question + final answer text) as history; do **not** carry
   evidence over, so a follow-up must call tools again and grounding stays as strict as now; a "new conversation"
   button; a multi-turn eval set (follow-ups that depend on the previous turn) added to the gate.
2. Larger eval set built from real user questions (39 questions give a 95% lower bound of 91%).
3. Retrieval experiments behind the gate: `multilingual-e5-large-instruct`, the `bge-reranker-v2-m3` cross-encoder
   (implemented, off by default).
4. Privacy hardening against differencing attacks on aggregates (query auditing, coarser filters).
5. Load test: questions per hour on one GPU with concurrent users.

## Data source
SF Employee Compensation: data.sfgov.org (dataset id `88g8-5mnd`), license PDDL 1.0.
Salary and benefits paid to City employees since fiscal year 2013. The CSV includes `Employee Name`,
which is treated as PII: it is hashed in staging and never reaches the marts or the agent.

## Acknowledgements
Capstone project for the Senior AI Engineering Series (rubythalib.ai). Patterns from the course labs
(vLLM serving, agentic RAG, grounding, MCP gateway, eval gate, observability) are re-implemented here for the HR domain.
