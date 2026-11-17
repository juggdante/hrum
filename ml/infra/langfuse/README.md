# Langfuse (self-hosted) for HRUM traces

Langfuse runs next to vLLM on the GPU PC, so traces stay on-premises like the model. The agent sends one trace
per question (LLM steps, tool calls, grounding) with personal data masked before it leaves the process
(`agent/observability/tracing.py`).

## 1. Start Langfuse (once)
Requires Docker with the compose plugin (`docker compose version`). Use the official compose file:

```bash
git clone --depth 1 https://github.com/langfuse/langfuse.git ~/langfuse
cd ~/langfuse
```

**Secrets.** Every `# CHANGEME` value in `docker-compose.yml` is `${VAR:-default}`, so set them in `~/langfuse/.env`
(read automatically by Docker Compose) instead of editing the file. The Postgres password appears twice
(`POSTGRES_PASSWORD`, inside `DATABASE_URL`) and the MinIO password four times (`MINIO_ROOT_PASSWORD` and the three
`LANGFUSE_S3_*_SECRET_ACCESS_KEY`); they must match. This writes random secrets without printing them:

```bash
gen() { openssl rand -hex 32; }
PG=$(gen); MINIO=$(gen)
cat > .env <<EOF
POSTGRES_PASSWORD=$PG
DATABASE_URL=postgresql://postgres:$PG@postgres:5432/postgres
SALT=$(gen)
ENCRYPTION_KEY=$(gen)
NEXTAUTH_SECRET=$(gen)
CLICKHOUSE_PASSWORD=$(gen)
REDIS_AUTH=$(gen)
MINIO_ROOT_PASSWORD=$MINIO
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$MINIO
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$MINIO
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$MINIO
EOF
chmod 600 .env && unset PG MINIO
```

**Ports.** The official file publishes the web UI (`3000`), MinIO (`9090`) and the AI gateway (`3031`) on all
interfaces, which on an office network exposes them to the whole LAN. Bind them to the PC itself:

```bash
sed -i 's|^      - 3000:3000$|      - 127.0.0.1:3000:3000|; s|^      - 9090:9000$|      - 127.0.0.1:9090:9000|; s|^      - \${LANGFUSE_AI_GATEWAY_PORT:-3031}:|      - 127.0.0.1:${LANGFUSE_AI_GATEWAY_PORT:-3031}:|' docker-compose.yml
grep -nA3 "ports:" docker-compose.yml | grep -- "- "    # every line should start with 127.0.0.1:
docker compose config --quiet && echo "compose config OK"
docker compose up -d
docker compose ps            # all services running/healthy after 1-3 minutes
```

The stack (web, worker, Postgres, ClickHouse, Redis, MinIO) needs a few GB of RAM and no GPU. If the official
instructions differ from the above (they change between releases), follow them: https://langfuse.com/self-hosting

## 2. Open the UI from your laptop
Langfuse listens on port 3000 of the GPU PC. From the laptop, either:
- VS Code Remote-SSH: the **Ports** tab forwards 3000 automatically (or "Forward a Port" → 3000), or
- `ssh -L 3000:127.0.0.1:3000 <gpu-pc>` and keep that session open.

Then browse to http://localhost:3000, sign up (the first user is local to this instance), create an
organisation and a project named `hrum`.

## 3. Connect HRUM
In the project: **Settings → API Keys → Create**. Put the keys in `hrum/.env` on the GPU PC:

```bash
LANGFUSE_HOST=http://127.0.0.1:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

`LANGFUSE_BASE_URL` (the name the Langfuse UI suggests) works too; set one of them. Quotes around values are
optional. Use `127.0.0.1` rather than `localhost`: the ports are bound to IPv4 only, and `localhost` may resolve
to IPv6 `::1` first. These keys go in the `.env` of the machine that runs the agent (the GPU PC); leave them
empty on the laptop.

Check, then use the agent as usual:

```bash
make trace-check                     # auth + one test trace (its input should show "[NAME]")
.venv/bin/python -m agent.cli --smoke
make eval ARGS="--only cost_01,docs_02,atk_01"
```

Tracing is off whenever the keys are empty, so tests and the laptop are unaffected.

## What a trace contains
| Observation | Type | Content |
|---|---|---|
| `answer` | agent | question (masked), final status / answer / claims / caveats, grounding, usage |
| `step_N`, `finalize`, `repair` | generation | messages, tool calls or text, token usage, latency |
| `tool:<name>` | tool / retriever | arguments, result (truncated), evidence id, errors |
| `grounding` | guardrail | claims checked, problems found |

Trace scores: `grounding_ok` (boolean), `repair_level` (0-3), `status` (answered / abstain / refused).
Tags: `role:<role>`, `prompt:<version>`, plus `cli`, `app` or `eval` + category. Eval runs share a
session id (`eval-<model>-<timestamp>`), so one session = one eval run; app chats get one session per browser tab.

## Masking
Applied by the SDK to every input, output and metadata value before export:
e-mail → `[EMAIL]`, phone → `[PHONE]`, 12+ digit ids → `[ID]`, 32-hex keys (hashed employee keys) → `[KEY]`,
and capitalised name-like words that are not in HRUM's own vocabulary (warehouse dimensions, metric catalogue,
knowledge base, prompts; never developer docs, which quote example names)
→ `[NAME]`. It over-masks rather than under-masks: an unfamiliar capitalised phrase can also become `[NAME]`.

## Without Docker
Langfuse Cloud works with the same three variables (`LANGFUSE_HOST=https://cloud.langfuse.com`), but traces then
leave the premises. Check company policy first; masking still applies.

## Troubleshooting
| Symptom | Check | Fix |
|---|---|---|
| UI keeps loading at http://localhost:3000 | On the GPU PC: `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000` | `200`/`307`: the server is fine and the VS Code port forward is stale (common after an SSH reconnect). Ports tab → port 3000 → **Stop Forwarding**, then forward it again. Also read the "Forwarded Address" column: 3000 may be mapped to another local port |
| `curl` hangs or returns `000` | `cd ~/langfuse && docker compose ps` | A service is not running/healthy: `docker compose logs --tail 40 langfuse-web` (or the failing service), then `docker compose up -d` |
| "tracing: Langfuse at … is not reachable; tracing off for this run" | `docker compose ps` | Langfuse is stopped while keys are in `.env`: start it (`docker compose up -d`), or empty the keys. HRUM keeps working without tracing |
| `make trace-check`: cannot reach Langfuse | `LANGFUSE_HOST` / `LANGFUSE_BASE_URL` in `hrum/.env` | Use `http://127.0.0.1:3000`; ports are bound to IPv4 only |
| `make trace-check`: keys rejected | Project → Settings → API Keys | Keys belong to one project; copy both again into `hrum/.env` |
| Langfuse uses RAM while idle | `docker stats --no-stream` | `docker compose stop` (data stays in the Docker volumes); `docker compose start` to resume. HRUM keeps working without it, traces are just not recorded |
