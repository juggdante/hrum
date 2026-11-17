# Setup guide: from `git clone` to a working chat

For someone who clones this repository onto a Linux PC with an NVIDIA GPU and wants the full stack running:
data warehouse → attrition model → document index → local LLM (vLLM + Qwen3.5) → Streamlit app, with optional
Langfuse tracing and the eval suite.

Run `make doctor` at any point. It checks every step below and prints the next command to run.

| Step | Command | First run takes | Needs GPU |
|---|---|---|---|
| 1. Prerequisites | (manual) | - | - |
| 2. Clone + `.env` | `git clone`, `cp .env.example .env` | 1 min | no |
| 3. App environment | `make setup` | 3-10 min | no |
| 4. Data warehouse | `make build-data` | 3-10 min (420 MB download) | no |
| 5. Attrition model | `make train` | 1-3 min | no |
| 6. Document index | `make index` | 1 min (downloads a 470 MB embedding model) | no |
| 7. Tests | `make test` | < 1 min | no |
| 8. LLM server | vLLM venv + `make serve` | 20-40 min (vLLM install, ~9 GB model, kernel compile) | yes |
| 9. App | `make agent` | 1 min | uses step 8 |
| 10. Optional | Langfuse, MCP, evals | see below | |

Steps 3-7 also work on a laptop without a GPU (use `make doctor ARGS=--cpu` there).

## 1. Prerequisites
- Linux with an NVIDIA GPU; `nvidia-smi` must work. The default profiles are tuned for 24 GB (RTX 4090). With
  less memory, lower `--gpu-memory-utilization` and `--max-model-len` in `configs/llm.yaml`.
- Python 3.11 or newer, `git`, and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Free disk: about 30 GB for the 4B model only, about 50 GB with the 9B candidate (model weights, two
  virtualenvs, data).
- Docker with the compose plugin, only for self-hosted Langfuse (step 10).
- A system CUDA toolkit (`nvcc`) is **not** required.

## 2. Clone and configure
```bash
git clone <this repository> hrum && cd hrum
cp .env.example .env
python3 -c "import secrets; print('QWEN_API_KEY=' + secrets.token_urlsafe(24))"
```
Put the printed `QWEN_API_KEY` line into `.env`. It is a password you invent for your own vLLM server; it does
not come from any provider. Leave the other values at their defaults for now.

Values are read from `.env` by the programs themselves. Do not `export` them in your shell, and leave VS Code's
`python.terminal.useEnvFile` off: a stale copy in the shell overrides later edits.

## 3. App environment
```bash
make setup                 # creates .venv with the exact versions in uv.lock (QUIET= shows the install log)
```
Expected: no output on success. Versions come from `uv.lock`, so you get exactly what the tests and evals ran with.
The app installs the CPU build of PyTorch (only the small embedding model uses it); the GPU is left to vLLM,
which has its own environment (step 8). `make doctor` then reports `app packages: all installed`.

## 4. Data warehouse (DuckDB + dbt)
```bash
make build-data
```
This downloads the SF Employee Compensation CSV (about 420 MB, public data), loads it into `data/hrum.duckdb`,
builds the staging views, generates the synthetic HRIS layer, then builds the marts and runs the dbt tests.
To reuse a CSV you already have, set `SF_COMP_CSV=/path/to/file.csv` in `.env` first.

Expected, near the end:
```
INFO Loaded 1,180,053 rows from Employee_Compensation_<date>.csv ...
INFO synthetic_hris: 40,344 employees, 14.6% terminated in FY2026
... Done. PASS=... WARN=0 ERROR=0
```
The source is updated by the City over time, so your row counts can differ from these. Everything downstream,
including the eval gold answers, is computed from your data, so tests and evals still apply.

## 5. Attrition model
```bash
make train
```
Trains XGBoost and a logistic-regression baseline on the synthetic FY2026 snapshot, writes the model card to
`ml/registry/attrition/` and publishes aggregated risk tables to `mart.ml_attrition_*`. Expected: a line with test
metrics (ROC-AUC around 0.72 for XGBoost) and `123 segments published` (with the current data).

## 6. Document index (RAG)
```bash
make index
```
Indexes `docs/` and the metric definitions with `intfloat/multilingual-e5-small` (downloaded from Hugging Face on
first use; set `HF_TOKEN` in `.env` for faster downloads).

## 7. Tests
```bash
make test
```
All tests run without a GPU (a scripted fake LLM replaces the model). They must all pass before you continue.

## 8. LLM server (GPU)
vLLM pins its own PyTorch build, so it lives in a separate virtualenv. `~/.venvs/vllm` is found automatically.
```bash
uv venv ~/.venvs/vllm --python 3.12
uv pip install --python ~/.venvs/vllm/bin/python "vllm==0.30.0"
make serve                                   # Qwen3.5-4B (bf16), the shipped model
```
The first start downloads the model (about 9 GB) and compiles GPU kernels; expect 15+ minutes, with progress
lines such as `[ 945s] compiling GPU kernels`. Later starts take about 30 seconds. It ends with
`vLLM ready in ...s at http://127.0.0.1:8000/v1`. Logs: `logs/vllm.log`.

Check it:
```bash
.venv/bin/python -m infra.local_server.vllm_server bench   # time to first token, tokens/s
.venv/bin/python -m agent.cli --smoke                      # one question end to end
```
Expected smoke result: `[answered] ...` with `grounding ok=True`.

Run long commands inside `tmux` when you work over SSH, so a dropped connection does not kill them
(`tmux new -s hrum`, detach with `Ctrl-b d`, reattach with `tmux attach -t hrum`).

## 9. The app
```bash
make agent                     # http://127.0.0.1:8501  (make agent APP_PORT=8502 if 8501 is taken)
```
The app listens on the PC itself only. From another machine, forward the port: VS Code Remote-SSH → **Ports** tab
→ Forward a Port → 8501, or `ssh -L 8501:127.0.0.1:8501 <gpu-pc>`. Then open http://localhost:8501.

Start with **Daftar pertanyaan** (question bank) and click **Tanya** on a few questions. Switch the role to
`analyst` in the sidebar to get the SQL Explorer page.

## 10. Optional
| What | How | Guide |
|---|---|---|
| Tracing (Langfuse, self-hosted) | Docker Compose next to vLLM, keys in `.env`, `make trace-check` | [infra/langfuse/README.md](../infra/langfuse/README.md) |
| MCP server (any MCP client) | set `HRUM_JWT_SECRET` (32+ chars) in `.env`, `make mcp` | [docs/governance.md](governance.md) |
| Eval + release gate | `make eval` (4B), `make serve PROFILE=qwen35_9b_fp8 && make eval` (9B), `make gate` | [README.md](../README.md#evaluation-and-release-gate) |

## Demo day
See [demo_runbook.md](demo_runbook.md): which terminal runs what, the ports to forward, `make demo-check`
(services up + one end-to-end question) and what to do if something fails mid-demo.

## If something fails
1. Run `make doctor` and follow its `->` hint.
2. vLLM did not start: the launcher prints the engine error lines; the full log is `logs/vllm.log`.
   Common causes and fixes are in [infra/local_server/README.md](../infra/local_server/README.md#troubleshooting).
3. Everything else we ran into while building this, with the fix, is in [docs/lessons_learned.md](lessons_learned.md).
