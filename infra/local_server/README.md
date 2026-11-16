# Local LLM server (RTX 4090 24 GB)

vLLM pins its own torch build, so it gets a separate virtualenv from the HRUM app.

```bash
# 1. vLLM venv (once). Version matches the course labs.
uv venv ~/.venvs/vllm --python 3.12 && uv pip install --python ~/.venvs/vllm/bin/python "vllm==0.30.0"
# ~/.venvs/vllm is found automatically; elsewhere: set VLLM_BIN in .env

# 2. HRUM app venv + data (once, CPU only)
make setup
SF_COMP_CSV=/path/to/Employee_Compensation.csv make build-data   # or leave unset to download

# 3. Serve a model (profiles in configs/llm.yaml). First start compiles kernels (several minutes).
make serve                          # Qwen/Qwen3.5-4B, bf16, 32k context
make serve PROFILE=qwen35_9b_fp8    # Qwen/Qwen3.5-9B, FP8 (candidate)

# 4. Check, then ask
.venv/bin/python -m infra.local_server.vllm_server status
.venv/bin/python -m infra.local_server.vllm_server bench
.venv/bin/python -m agent.cli --smoke
make agent                          # Streamlit chat
```

Set `QWEN_API_KEY` in `.env` (it is passed to `vllm serve --api-key` and used by the client).
Only one model is on the GPU at a time: `vllm_server start` stops the previous one first.
Logs: `logs/vllm.log`.

| Profile | Model | Weights | Context | GPU util |
|---|---|---|---|---|
| `qwen35_4b` | Qwen/Qwen3.5-4B | bf16 (~9 GB) | 32,768 | 0.80 |
| `qwen35_9b_fp8` | Qwen/Qwen3.5-9B | FP8 on-the-fly (~10 GB) | 16,384 | 0.85 |
| `qwen35_4b_t4` | Qwen/Qwen3.5-4B | fp16 (Colab T4 fallback) | 16,384 | 0.85 |

## MCP server
```bash
# .env: HRUM_JWT_SECRET=<64 hex chars>
make mcp                                                  # http://127.0.0.1:8765/mcp, JWT required
.venv/bin/python -m agent.mcp.server --mint-token chro    # demo bearer token (role chro or analyst)
make mcp TRANSPORT=stdio                                  # for a local MCP client; role from HRUM_MCP_ROLE
```
Example stdio entry for an MCP client config:
```json
{"mcpServers": {"hrum": {"command": "/path/to/hrum/.venv/bin/python",
  "args": ["-m", "agent.mcp.server", "--transport", "stdio"],
  "cwd": "/path/to/hrum", "env": {"HRUM_MCP_ROLE": "chro"}}}}
```

## Troubleshooting
- **`nvcc fatal: Unknown option '--compress-mode=size'`** (Ninja build failed in FlashInfer sampling):
  the system `nvcc` is older than FlashInfer expects. The launcher sets `VLLM_USE_FLASHINFER_SAMPLER=0`
  so vLLM uses its PyTorch sampler and compiles nothing. Delete `~/.cache/flashinfer` after a failed build.
- **`Engine core initialization failed`**: that line is only the API server's summary; the real error is in
  the `EngineCore` lines of `logs/vllm.log` (the launcher prints them on failure).
- **CUDA out of memory**: lower `--gpu-memory-utilization` or `--max-model-len` in the profile, or add `--enforce-eager`.
