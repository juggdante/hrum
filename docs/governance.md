# Governance & Limitations
- Real data: SF public employee compensation (PDDL). Synthetic data: clearly labelled `is_synthetic`.
- Agent: read-only, mart layer only, min group size 5, no individual lookups.
- Models trained on synthetic fields demonstrate capability, not findings.
- Every agent answer must show: metric definition used, SQL/tool call, caveats.
- `Employee Name` exists in the source CSV. Treat as PII: stays in raw/staging only; marts use a hashed surrogate key (`employee_key`) or drop it. Never load it into agent-visible datasets or prompts.

## Agent guardrails (implemented)
| Layer | Control | Code |
|---|---|---|
| Access | CHRO role: governed metrics only (`query_metric`, `calculate`). Analyst role adds `run_sql` | `agent/tools/` (Tool.roles) |
| SQL | sqlglot AST guard: one SELECT, `mart.*` only, no DDL/DML/ATTACH/COPY/PRAGMA, no file-reading table functions, no `SELECT *`, no person-level keys, outer SELECT must aggregate; rewritten with `HAVING count(*) >= 5` and `LIMIT 1000` | `agent/guardrails/sql_guard.py` |
| Privacy | Semantic layer adds `HAVING count(*) >= 5` to every metric query; names never reach `mart` | `common/semantic.py` |
| Execution | DuckDB opened `read_only`, query timeout | `agent/core/context.py`, `agent/tools/sql.py` |
| Answers | Every number must match cited evidence; one repair, then unsupported claims are dropped, else abstain. Caveats and SQL are attached by code | `agent/guardrails/grounding.py`, `agent/core/loop.py` |

Residual risk: an analyst could craft sub-queries that narrow a group before the outer aggregate.
Mitigation: run_sql is analyst-only, all calls are traced, and the eval attack suite covers it.

## MCP gateway (implemented, `agent/mcp/`)
HRUM tools are also exposed to any MCP client through the official MCP SDK. Every call passes the gateway:

| Control | What it stops | Check |
|---|---|---|
| JWT (HS256, audience + issuer + expiry) | Tokens minted for another service, forged or expired tokens (HTTP 401 before any tool runs) | `verify_token` |
| Scope → role (`hrum.chro`, `hrum.analyst`) | CHRO clients calling `run_sql` | `role_from_scopes`, `Tool.roles` |
| Allowlist | Calls to tools that are not HRUM tools (shadowing by name) | `Gateway.call` |
| Fingerprint pinning | A tool definition changed after approval (rug pull); pins in `agent/mcp/tool_pins.json` | `fingerprint` |
| Definition scan | Descriptions carrying prompt-injection text (tool poisoning) | `scan_definitions` |
| Rate limit | More than 60 calls/minute per client | `Controls` |
| Output wrapping | Tool output being treated as instructions | `OUTPUT_NOTICE` |
| Audit log | Untraceable access; every allow/deny/error in `logs/mcp_audit.jsonl` (no result rows) | `_record` |

Transports: Streamable HTTP on `127.0.0.1:8765/mcp` (JWT required) and stdio for a local trusted client, whose role
comes from `HRUM_MCP_ROLE`. All tools are read-only, so no human-in-the-loop approval step is needed.
After reviewing an intended change to a tool definition, re-approve with `python -m agent.mcp.server --update-pins`.

## Observability and privacy (implemented, `agent/observability/`)
Each question becomes one Langfuse trace (self-hosted on the GPU PC, `infra/langfuse/README.md`): the agent run,
every LLM call with tokens and latency, every tool call, and the grounding check, scored with `grounding_ok`,
`repair_level` and `status`. Tracing is off unless Langfuse keys are configured and never changes an answer.

| Control | What it stops | Where |
|---|---|---|
| Mask before export | Names, e-mails, phone numbers, long ids and hashed employee keys reaching the trace store | `tracing.mask` (SDK `mask` hook) |
| Name rule = vocabulary check | A person's name typed into a question (the warehouse holds none); unknown capitalised phrases become `[NAME]` | `tracing._mask_names` |
| Self-hosted store | Traces leaving the premises | `infra/langfuse/` |
| Aggregated evidence only | Person-level rows in traces: tools return aggregates (groups of at least 5) | SQL guard, semantic layer |

Residual risk: the name rule is heuristic. A name made only of words that also occur in HRUM's vocabulary (for
example a surname that is also a department word) would not be masked.

## SQL Explorer page (implemented, `agent/app/views/sql_explorer.py`)
A page in the HRUM app where a person with the analyst role writes SQL directly. Decision (2026-11-12): unlike the
agent's `run_sql`, it may return **row-level pay records**, because the person sees the query they wrote and the
source is public data. It keeps every other control (`check_sql(policy="explore")`):

| Control | Explore page | Agent `run_sql` |
|---|---|---|
| One read-only SELECT on schema `mart`, no file/system functions | yes | yes |
| Person keys (`employee_key`, `compensation_record_id`) | blocked; `SELECT *` rewritten to `* EXCLUDE (...)` | blocked; `SELECT *` rejected |
| Outer SELECT must aggregate, `HAVING COUNT(*) >= 5` | no | yes |
| Row limit / timeout | 5,000 rows / 20 s | 1,000 rows / 20 s |
| Visible to | analyst role only (page not registered for CHRO) | analyst role only |

Residual risk: a record can be narrowed by job title and department to one person, whose pay (public in the SF
source) is then shown without the name. Accepted for a human analyst tool; agent answers stay aggregated.
`employee_key` stays blocked because it is the md5 of a fixed prefix plus the name, so hashing a known name
would find that person's rows.
