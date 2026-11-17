"""Start / wait / stop / benchmark the local vLLM server (RTX 4090 or Colab T4).

Adapted from course labs P2 (`start_server`, `wait_ready`, `bench`, `ttft_only`) and P10 (launch flags,
startup phases). Launch profiles live in configs/llm.yaml `serve_profiles`.

    python -m infra.local_server.vllm_server start qwen35_4b
    python -m infra.local_server.vllm_server status | stop | bench
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

from common.config import ROOT, load_config

HOST, PORT = "127.0.0.1", int(os.environ.get("VLLM_PORT", "8000"))
BASE = f"http://{HOST}:{PORT}"
LOG_DIR = ROOT / "logs"
LOG, PID = LOG_DIR / "vllm.log", LOG_DIR / "vllm.pid"
# vLLM pins torch, so it lives in its own venv. Order: VLLM_BIN (shell or .env), the documented venv, PATH.
DEFAULT_VLLM = Path.home() / ".venvs" / "vllm" / "bin" / "vllm"
VLLM_BIN = os.path.expanduser(os.environ.get("VLLM_BIN", "")) or (
    str(DEFAULT_VLLM) if DEFAULT_VLLM.exists() else "vllm")
PHASES = [("Loading weights", "loading weights to GPU"), ("Memory profiling", "profiling GPU memory"),
          ("autotun", "compiling GPU kernels (slowest step)"), ("Compiling", "compiling GPU kernels (slowest step)"),
          ("CUDA graph", "capturing CUDA graphs"), ("Capturing", "capturing CUDA graphs"),
          ("Starting vLLM API server", "starting API server")]
FAILURES = ("Engine core initialization failed", "CUDA out of memory", "Traceback")


def build_command(profile: str) -> list[str]:
    cfg = load_config("llm")
    profiles = cfg["serve_profiles"]
    if profile not in profiles or profile == "common_args":
        raise SystemExit(f"Unknown profile '{profile}'. Options: {[p for p in profiles if p != 'common_args']}")
    qwen = cfg["providers"]["qwen_local"]
    return [VLLM_BIN, "serve", profiles[profile]["hf_model"], "--served-model-name", qwen["model"],
            "--host", HOST, "--port", str(PORT), "--api-key", qwen["api_key"],
            *profiles[profile]["args"], *profiles["common_args"]]


def is_ready() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def current_phase() -> str:
    text = LOG.read_text(errors="ignore") if LOG.exists() else ""
    found = [label for key, label in PHASES if key in text]
    return found[-1] if found else "starting"


def stop() -> None:
    """Stop any running vLLM server (same patterns as the course labs)."""
    subprocess.run("pkill -f '[v]llm serve'; sleep 3; pkill -9 -f '[v]llm serve'; "
                   "pkill -9 -f '[V]LLM::EngineCore'", shell=True, check=False)
    PID.unlink(missing_ok=True)


def serve_env() -> dict[str, str]:
    """Environment for `vllm serve`.

    FlashInfer's sampler JIT-compiles CUDA kernels with the system nvcc on first use; an nvcc older than
    the one FlashInfer expects fails (e.g. "Unknown option '--compress-mode=size'"). The PyTorch sampler
    needs no compilation and is fast enough here. Export VLLM_USE_FLASHINFER_SAMPLER=1 to opt back in.
    """
    env = dict(os.environ)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


def start(profile: str) -> subprocess.Popen:
    if shutil.which(VLLM_BIN) is None:
        raise SystemExit(f"vLLM not found at '{VLLM_BIN}'. Create its venv first (infra/local_server/README.md step 1):\n"
                         '  uv venv ~/.venvs/vllm --python 3.12 && '
                         'uv pip install --python ~/.venvs/vllm/bin/python "vllm==0.30.0"\n'
                         "~/.venvs/vllm is found automatically; for another location set VLLM_BIN in .env")
    stop()
    LOG_DIR.mkdir(exist_ok=True)
    cmd = build_command(profile)
    print("$", " ".join(shlex.quote(c) for c in cmd).replace(load_config("llm")["providers"]["qwen_local"]["api_key"], "***"))
    with open(LOG, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=serve_env())
    PID.write_text(str(proc.pid))
    return proc


def root_cause(text: str, n: int = 25) -> str:
    """Error lines from the engine process; the API server only reports 'Engine core initialization failed'."""
    keys = ("Error", "error:", "out of memory", "OOM", "Exception")
    lines = [ln for ln in text.splitlines() if "APIServer" not in ln and any(k in ln for k in keys)]
    return "\n".join(lines[-n:] or text.splitlines()[-n:])


def wait_ready(timeout: int = 1800, every: int = 15) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_ready():
            print(f"vLLM ready in {time.time() - t0:.0f}s at {BASE}/v1")
            return True
        text = LOG.read_text(errors="ignore") if LOG.exists() else ""
        if any(f in text for f in FAILURES[:2]):
            print("vLLM failed to start. Root cause (engine log):\n" + root_cause(text))
            print(f"Full log: {LOG}")
            return False
        print(f"[{time.time() - t0:5.0f}s] {current_phase()}", flush=True)
        time.sleep(every)
    print("Timed out waiting for vLLM")
    return False


def bench(n: int = 8, max_tokens: int = 256) -> dict:
    """Sequential latency benchmark: time-to-first-token and decode tokens/s via streaming."""
    from openai import OpenAI

    qwen = load_config("llm")["providers"]["qwen_local"]
    client = OpenAI(base_url=f"{BASE}/v1", api_key=qwen["api_key"])
    ttfts, tps = [], []
    for i in range(n):
        t0, first, chunks = time.time(), None, 0
        stream = client.chat.completions.create(
            model=qwen["model"], max_tokens=max_tokens, temperature=0, stream=True,
            messages=[{"role": "user", "content": f"Explain HR turnover in {80 + i} words."}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        for ch in stream:
            if ch.choices and ch.choices[0].delta.content:
                first = first or time.time()
                chunks += 1
        if first:
            ttfts.append(first - t0)
            tps.append(chunks / max(time.time() - first, 1e-6))
    ttfts.sort()
    result = {"requests": n, "ttft_p50_s": round(ttfts[len(ttfts) // 2], 3),
              "ttft_max_s": round(ttfts[-1], 3), "decode_tokens_per_s": round(sum(tps) / len(tps), 1)}
    print(result)
    return result


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "status"
    if cmd == "start":
        start(argv[1] if len(argv) > 1 else "qwen35_4b")
        sys.exit(0 if wait_ready() else 1)
    elif cmd == "stop":
        stop()
    elif cmd == "bench":
        bench()
    else:
        print(f"ready={is_ready()} phase={current_phase()} log={Path(LOG)}")


if __name__ == "__main__":
    main(sys.argv[1:])
