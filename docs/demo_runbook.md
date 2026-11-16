# Demo runbook

How to bring HRUM up for a live demo, check it, run the demo and recover if something fails.
First-time installation is in [setup_guide.md](setup_guide.md); this page assumes setup is done.

## Services at a glance
| Service | Start | Port | Needed | Keeps its terminal? |
|---|---|---|---|---|
| vLLM (Qwen3.5-4B) | `make serve` | 8000 | **yes** | no: returns when ready, keeps running in the background |
| HRUM app (Streamlit) | `make agent APP_PORT=8502` | 8501 / 8502 | **yes** | **yes**: stop with Ctrl-C |
| Langfuse (tracing) | `cd ~/langfuse && docker compose up -d` | 3000 | optional | no: Docker runs it in the background |
| MCP server | `make mcp` | 8765 | optional | **yes** |

All of them listen on the GPU PC only (`127.0.0.1`). From the laptop, forward ports in VS Code (Ports tab) or
with `ssh -L`. If the app port is taken by someone else, use another one (`APP_PORT=8503`) everywhere below.

**vLLM keeps running after `make serve`.** It runs as its own background process, so it survives a closed
terminal or a dropped SSH session. It stops only with `vllm_server stop`, a reboot or a crash. While it runs it
holds about 80% of the GPU memory (`--gpu-memory-utilization 0.80`) even when idle, so on a shared PC stop it after
the demo (see the last section).

## Is it running? Checking services and ports
```bash
# which HRUM ports are listening: vLLM 8000, app 8501/8502, Langfuse 3000, MCP 8765
ss -ltnp | grep -E ':(8000|8501|8502|3000|8765)\b'

# is each service healthy (not just a port in use)?
curl -s -o /dev/null -w "vLLM      %{http_code}\n" http://127.0.0.1:8000/health
curl -s -w "  <- app 8502\n" http://127.0.0.1:8502/_stcore/health
curl -s -o /dev/null -w "Langfuse  %{http_code}\n" http://127.0.0.1:3000

.venv/bin/python -m infra.local_server.vllm_server status   # vLLM ready=True/False and its log file
nvidia-smi                                                   # who is using GPU memory
cd ~/langfuse && docker compose ps                           # Langfuse containers
```
- `200` (vLLM), `ok` (app) and `200`/`307` (Langfuse) mean healthy.
- `ss` shows the process name (`users:((...))`) only for your own processes. A port listed without one belongs
  to another user or to Docker. On a shared PC, do not stop processes you do not own.
- `make demo-check APP_PORT=8502` runs all of these checks at once, plus one real question.

## Terminal layout on the GPU PC
Use `tmux` from a plain SSH terminal so a dropped connection does not stop the app (inside the VS Code terminal,
VS Code takes `Ctrl-b`; there, just open separate VS Code terminals instead).

```
Terminal 1  (services, then free)      Terminal 2  (app, stays open)     Terminal 3  (checks)
-------------------------------------  --------------------------------  ---------------------------------
cd ~/langfuse && docker compose up -d  cd <repo>                         cd <repo>
cd <repo> && make serve                make agent APP_PORT=8502          make demo-check APP_PORT=8502
```
Optional Terminal 4: `make mcp` (only if you show an MCP client).

## Checklist
**30 minutes before**
1. `git pull` on the GPU PC; `make setup` if dependencies changed.
2. Start the services (Terminal 1 and 2 above). vLLM takes ~30 seconds when the model is cached.
3. Laptop: forward the app port (8502) and, if you show tracing, 3000. If a page just keeps loading, the forward is
   stale: Ports tab → Stop Forwarding → forward again.

**10 minutes before**
4. `make demo-check` (or `make demo-check APP_PORT=8502`). Without a port it finds this repository's app by its
   process, so another Streamlit app on 8501 is not mistaken for HRUM. It must end with `Ready for the demo.` It checks the data, model, index, vLLM,
   the app's health endpoint, Langfuse (containers and API, if configured), MCP, and runs **one real question end to
   end** (this also warms up the model). Fix every `FAIL` using the `->` hint; a `WARN` on an optional service means
   that part of the demo is unavailable.
5. Open the browser tabs: the app (Tanya HRUM, Daftar pertanyaan, Panduan dataset) and, optionally, Langfuse →
   Tracing (filter *Is Root Observation = True*) or Sessions.
6. Ask one question in the app so the page and charts are warm.

## Suggested demo flow (10–15 minutes)
| Step | What to do | What to point out |
|---|---|---|
| 1 | Daftar pertanyaan → **Tanya** on cost_03 (total compensation FY2022–2026) | Line chart drawn from the cited rows; SQL and metric definition in the evidence panel |
| 2 | cost_06 (department with the highest overtime) | Sorted bar chart; real data |
| 3 | risk_05 (department with the highest attrition risk) | KPI tiles; SYNTHETIC caveat added by the system |
| 4 | Ask "Berapa gaji Stephen Lee?" | Not answered (privacy rule) |
| 5 | Langfuse: open that trace | `[NAME]` in every span; latency per step; scores |
| 6 | Sidebar role → analyst; SQL Explorer → an example query | `SELECT *` rewritten to exclude person keys |
| 7 | Panduan dataset | What is real vs synthetic; metric catalogue |
| 8 | Back to the deck: eval and release gate | 39/39, 0/8 breaches, why 4B ships |

## If something fails during the demo
| Symptom | Quick fix | Fallback |
|---|---|---|
| App says "Model tidak bisa dihubungi" | Terminal 1: `make serve` (~30 s) | Deck slides 6–11 |
| Page keeps loading | Re-forward the port (stale forward) | Refresh after re-forwarding |
| First answer slow (10+ s) | Normal once after a restart (model warm-up) | Talk through the evidence panel meanwhile |
| Langfuse not reachable | `cd ~/langfuse && docker compose up -d` | Skip step 5; the app keeps working without tracing |
| "Port is not available" for the app | `make agent APP_PORT=8503` and forward 8503 | |
| SSH dropped | Services survive; restart the app if it was not in tmux | |

## After the demo
```bash
.venv/bin/python -m infra.local_server.vllm_server stop     # free the GPU
cd ~/langfuse && docker compose stop                        # data stays in the Docker volumes
```
Stop the app with Ctrl-C in its terminal.
