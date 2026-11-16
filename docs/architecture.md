# Architecture

HRUM runs entirely on one Linux PC with an NVIDIA GPU (developed on an RTX 4090, 24 GB). Nothing is sent to a cloud
service: the language model, the warehouse, the document index and the tracing store are all local.

## Components and data flow
```
SF Employee Compensation CSV (REAL, public) ─┐
Synthetic HRIS generator (SYNTHETIC) ────────┴─► DuckDB raw ─► dbt staging ─► dbt marts ─► semantic layer (16 metrics)
                                                                               │                │
                                         attrition model (XGBoost + SHAP) ◄────┘                │
                                         publishes aggregated risk tables only                  │
HR knowledge base, data dictionary, governance, metric definitions ─► RAG index (BM25 + e5)     │
                                                                               │                │
                                                                               ▼                ▼
                          Qwen3.5-4B on vLLM  ◄──►  agent loop: tools, budgets, SQL guard, grounding
                                                                               │
                         ┌─────────────────────────────┬───────────────────────┴──────────────┐
                    Streamlit app                 MCP gateway                         Langfuse (self-hosted)
             chat + charts, question bank,   other AI clients, JWT roles,          traces per question,
             dataset guide, SQL Explorer     tool pinning, audit log               personal data masked
```

| Layer | What it does | Where |
|---|---|---|
| Ingestion | Loads the CSV as-is (all text columns); generates the synthetic HRIS layer deterministically | `ingestion/` |
| Warehouse | DuckDB file `data/hrum.duckdb`, schemas `raw` → `stg` → `mart`, dbt models with tests | `warehouse/` |
| Semantic layer | One definition per metric; compiles metric + dimensions + filters into parameterised SQL with a minimum group size of 5 | `warehouse/semantic/metrics.yaml`, `common/semantic.py` |
| ML | Attrition model trained on the synthetic snapshot; only segment-level results reach the mart | `ml/` |
| Retrieval | Heading-aware chunks of the knowledge base and metric definitions; BM25 and multilingual-e5 fused with RRF | `agent/rag/`, `configs/rag.yaml` |
| LLM serving | vLLM serves Qwen3.5-4B (bf16) behind an OpenAI-compatible API; a 9B candidate exists for the release gate | `infra/local_server/`, `configs/llm.yaml` |
| Agent | Tool-calling loop with budgets; answers are JSON claims citing evidence and are checked by the grounding validator | `agent/core/`, `agent/guardrails/`, `agent/tools/` |
| Interfaces | Streamlit app, command line, MCP server | `agent/app/`, `agent/cli.py`, `agent/mcp/` |
| Observability | One Langfuse trace per question; masking before export | `agent/observability/`, `infra/langfuse/` |
| Evaluation | Golden questions with gold answers computed by SQL; release gate with exact McNemar test | `evals/` |

## How one question is answered
1. The user asks in the app (or CLI / MCP). The role (CHRO or analyst) decides which tools are available.
2. The model chooses tools. Each tool result is stored as evidence (E1, E2, …):
   `query_metric` (numbers from the semantic layer), `search_docs` (definitions and rules), `attrition_insight`
   (aggregated model output), `calculate` (arithmetic on evidence numbers only), `run_sql` (analyst only, guarded).
3. The model writes the final answer as JSON; every factual claim cites evidence ids.
4. The grounding validator checks that every number appears in the cited evidence. If not, the model gets one repair
   attempt; unsupported claims are then dropped, and if nothing verifiable remains the answer is "abstain".
5. The app shows the answer, a chart drawn by code from the cited rows, caveats (for example SYNTHETIC data) and the
   evidence: metric definitions, the SQL that ran and its rows.

## Runtime topology
| Process | Port (127.0.0.1 only) | Started by |
|---|---|---|
| vLLM model server | 8000 | `make serve` |
| Streamlit app | 8501 (or `APP_PORT`) | `make agent` |
| Langfuse (Docker Compose) | 3000 | `docker compose up -d` in the Langfuse folder |
| MCP server (optional) | 8765 | `make mcp` |

Remote users reach the app through SSH port forwarding. `make doctor` checks the whole stack; `make demo-check` also
asks one real question end to end.

## Design principles
- **Guarantees live in code, not in the prompt:** numbers come from the semantic layer, SQL is checked by an AST
  guard, privacy limits are enforced in queries, charts are drawn from evidence.
- **Real and synthetic data are always labelled;** results on synthetic fields demonstrate the method, not findings.
- **Every model or prompt change goes through the eval release gate;** the smaller model stays unless a larger one is
  significantly better.

More detail: `docs/governance.md` (controls), `docs/data_dictionary.md` (data), `README.md` (setup and evaluation).
