"""`make doctor`: check every prerequisite and setup step, and say what to run next.

    make doctor              # GPU machine: full check
    make doctor ARGS=--cpu   # laptop / CI: GPU, vLLM and model-server checks are skipped
    make demo-check          # before a demo: services must be up, plus one end-to-end question

Runs with only the standard library at the top level, so it works before `make setup`; checks that need the
project's packages report "run make setup" instead of crashing. Exit code 1 if any check FAILs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "bin" / "python"
OK, WARN, FAIL, SKIP, INFO = "OK", "WARN", "FAIL", "SKIP", "INFO"
MCP_PORT = 8765
# set by main(): cpu (skip GPU checks), demo (services required + smoke question), app_port
OPTS = argparse.Namespace(cpu=False, demo=False, app_port=None, smoke=True)


@dataclass
class Result:
    status: str
    name: str
    detail: str = ""
    fix: str = ""


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def read_env() -> dict[str, str]:
    """Minimal .env reader (KEY=value, optional quotes, trailing ' # comment'); the shell env wins."""
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip().strip("'\"")
            values[key.strip()] = val
    return {**values, **{k: v for k, v in os.environ.items() if k in values}}


def _venv_python(code: str, timeout: int = 60) -> tuple[int, str]:
    if not VENV_PY.exists():
        return 127, "no .venv"
    return _run([str(VENV_PY), "-c", code], timeout)


# ------------------------------------------------------------------ checks

def check_tools(cpu: bool = False) -> list[Result]:
    out = []
    v = sys.version_info
    out.append(Result(OK if v >= (3, 10) else FAIL, "python", f"{v.major}.{v.minor}", "install Python >= 3.10"))
    for tool, fix in (("uv", "curl -LsSf https://astral.sh/uv/install.sh | sh"), ("git", "install git")):
        path = shutil.which(tool)
        out.append(Result(OK if path else FAIL, tool, path or "not found", fix))
    free = shutil.disk_usage(Path.home()).free / 1e9
    if cpu:   # no model weights on this machine: the app venv and data need ~5 GB
        out.append(Result(OK if free >= 5 else WARN, "disk free (home)", f"{free:.0f} GB", "free ~5 GB"))
    else:
        out.append(Result(OK if free >= 50 else WARN if free >= 30 else FAIL, "disk free (home)", f"{free:.0f} GB",
                          "models + venvs need ~30 GB (4B only) to ~50 GB (4B + 9B)"))
    return out


def check_env(cpu: bool = False) -> list[Result]:
    env = read_env()
    if not (ROOT / ".env").exists():
        return [Result(WARN if cpu else FAIL, ".env", "missing", "cp .env.example .env, then set QWEN_API_KEY")]
    out = [Result(OK, ".env", "present")]
    key = env.get("QWEN_API_KEY", "")
    out.append(Result(OK if key and key != "change-me" else WARN, "QWEN_API_KEY",
                      "set" if key and key != "change-me" else "default 'change-me'",
                      'python3 -c "import secrets; print(secrets.token_urlsafe(24))"  -> .env'))
    jwt = env.get("HRUM_JWT_SECRET", "")
    out.append(Result(OK if len(jwt) >= 32 else INFO, "HRUM_JWT_SECRET (MCP only)",
                      "set" if len(jwt) >= 32 else "not set: needed only for `make mcp`",
                      'python3 -c "import secrets; print(secrets.token_hex(32))"  -> .env'))
    lf = bool(env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"))
    out.append(Result(INFO, "Langfuse keys (optional)", "set: tracing on" if lf else "empty: tracing off",
                      "infra/langfuse/README.md"))
    return out


def check_venv() -> list[Result]:
    if not VENV_PY.exists():
        return [Result(FAIL, "app venv (.venv)", "missing", "make setup")]
    mods = ["duckdb", "dbt.cli", "sqlglot", "openai", "sentence_transformers", "xgboost", "shap", "mlflow",
            "streamlit", "plotly", "mcp", "jwt", "langfuse"]
    code = (f"import importlib.util as u; print(','.join(m for m in {mods!r} "
            "if u.find_spec(m.split('.')[0]) is None))")
    rc, missing = _venv_python(code)
    missing = missing.strip() if rc == 0 else "?"
    return [Result(OK if rc == 0 and not missing else FAIL, "app packages",
                   "all installed" if not missing else f"missing: {missing}", "make setup")]


def check_gpu() -> list[Result]:
    rc, out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                    "--format=csv,noheader,nounits"])
    if rc != 0:
        return [Result(FAIL, "NVIDIA GPU", "nvidia-smi not available", "install the NVIDIA driver")]
    name, total, used, driver = [x.strip() for x in out.splitlines()[0].split(",")]
    total_gb, free_gb = int(total) / 1024, (int(total) - int(used)) / 1024
    res = [Result(OK if total_gb >= 20 else WARN, "NVIDIA GPU", f"{name}, {total_gb:.0f} GB, driver {driver}",
                  "profiles assume 24 GB; lower --gpu-memory-utilization / --max-model-len in configs/llm.yaml")]
    res.append(Result(OK if free_gb >= 0.8 * total_gb or _port_open(8000) else WARN, "GPU memory free",
                      f"{free_gb:.1f} GB free", "another process uses the GPU: `nvidia-smi`"))
    rc, out = _run(["nvcc", "--version"])
    rel = re.search(r"release ([\d.]+)", out)
    res.append(Result(INFO, "nvcc (system CUDA toolkit)", rel.group(1) if rc == 0 and rel else "not found",
                      "not required: the launcher uses vLLM's PyTorch sampler (VLLM_USE_FLASHINFER_SAMPLER=0)"))
    return res


def _vllm_bin(env: dict[str, str]) -> str | None:
    explicit = os.path.expanduser(env.get("VLLM_BIN", ""))
    default = Path.home() / ".venvs" / "vllm" / "bin" / "vllm"
    for cand in (explicit, str(default) if default.exists() else "", shutil.which("vllm") or ""):
        if cand and Path(cand).exists():
            return cand
    return None


def check_vllm() -> list[Result]:
    env = read_env()
    binary = _vllm_bin(env)
    if not binary:
        return [Result(FAIL, "vLLM", "not installed",
                       'uv venv ~/.venvs/vllm --python 3.12 && uv pip install --python ~/.venvs/vllm/bin/python '
                       '"vllm==0.30.0"')]
    py = Path(binary).parent / "python"
    rc, ver = _run([str(py), "-c", "import importlib.metadata as m; print(m.version('vllm'))"])
    res = [Result(OK, "vLLM", f"{ver if rc == 0 else '?'} at {binary}")]
    res.append(Result(INFO, "docker (Langfuse only)", "available" if shutil.which("docker") else "not found",
                      "needed only for self-hosted Langfuse"))
    return res


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_data() -> list[Result]:
    if not VENV_PY.exists():
        return [Result(SKIP, "warehouse", "needs .venv", "make setup")]
    code = """
import json, duckdb
from common.config import db_path
p = db_path()
if not p.exists():
    print(json.dumps({"db": None})); raise SystemExit
con = duckdb.connect(str(p), read_only=True)
have = {f"{s}.{t}" for s, t in con.execute("select table_schema, table_name from information_schema.tables").fetchall()}
want = ["raw.sf_employee_compensation", "raw.synthetic_hris", "mart.fct_compensation", "mart.fct_employee_snapshot",
        "mart.ml_attrition_risk_segment"]
print(json.dumps({"db": str(p), "tables": {t: (con.execute(f"select count(*) from {t}").fetchone()[0]
                                               if t in have else None) for t in want}}))
"""
    rc, out = _run([str(VENV_PY), "-c", code], 60)
    if rc != 0:
        return [Result(FAIL, "warehouse", out.splitlines()[-1][:200] if out else "error", "make build-data")]
    info = json.loads(out.splitlines()[-1])
    if not info["db"]:
        return [Result(FAIL, "warehouse (DuckDB)", "data/hrum.duckdb missing", "make build-data")]
    t = info["tables"]
    fixes = {"raw.sf_employee_compensation": "make ingest (or make build-data)",
             "raw.synthetic_hris": "make synth", "mart.fct_compensation": "make dbt-run && make dbt-test",
             "mart.fct_employee_snapshot": "make dbt-run && make dbt-test", "mart.ml_attrition_risk_segment": "make train"}
    return [Result(OK if n else FAIL, name, f"{n:,} rows" if n else "missing", fixes[name]) for name, n in t.items()]


def check_artifacts() -> list[Result]:
    out = []
    model = ROOT / "ml" / "registry" / "attrition" / "model.json"
    out.append(Result(OK if model.exists() else FAIL, "attrition model", "present" if model.exists() else "missing",
                      "make train"))
    idx = ROOT / "data" / "index"
    if not (idx / "chunks.parquet").exists():
        out.append(Result(FAIL, "RAG index", "missing", "make index"))
    else:
        rc, cfg_model = _venv_python("from common.config import load_config; "
                                     "print(load_config('rag')['embedding']['model'])")
        built = (idx / "model.txt").read_text().strip() if (idx / "model.txt").exists() else "?"
        if rc != 0:
            out.append(Result(SKIP, "RAG index", f"built with {built}; config check needs .venv", "make setup"))
        else:
            same = cfg_model.splitlines()[-1].strip() == built
            out.append(Result(OK if same else WARN, "RAG index", f"built with {built}",
                              "embedding model changed: make index"))
    return out


def http_ok(url: str, timeout: float = 3) -> tuple[bool, str]:
    """(True, body) when the URL answers 200."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # local health endpoints only
            return r.status == 200, r.read(200).decode(errors="ignore").strip()
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, type(e).__name__


def parse_cli_json(out: str) -> dict | None:
    """The `agent.cli --json` result: the JSON object in the output (other lines may follow, e.g. tracing)."""
    start, end = out.find("{"), out.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(out[start:end + 1])
    except json.JSONDecodeError:
        return None


def langfuse_containers() -> tuple[int, int, list[str]] | None:
    """(running, total, not-ok names) of Docker containers named langfuse-*; None when Docker is unavailable."""
    if not shutil.which("docker"):
        return None
    rc, out = _run(["docker", "ps", "-a", "--filter", "name=langfuse", "--format", "{{.Names}}\t{{.Status}}"])
    if rc != 0:
        return None
    rows = [ln.split("\t", 1) for ln in out.splitlines() if "\t" in ln]
    bad = [n for n, st in rows if not st.startswith("Up") or "unhealthy" in st]
    return len(rows) - len(bad), len(rows), bad


_SS_LINE = re.compile(r":(\d+)\s+\S+\s+users:\(\(\"([^\"]+)\",pid=(\d+)")


def listeners() -> dict[int, list[int]]:
    """port -> pids listening on it. `ss` shows pids only for processes this user may inspect."""
    rc, out = _run(["ss", "-ltnpH"])
    return parse_listeners(out if rc == 0 else "")


def parse_listeners(text: str) -> dict[int, list[int]]:
    found: dict[int, list[int]] = {}
    for m in _SS_LINE.finditer(text):
        found.setdefault(int(m.group(1)), []).append(int(m.group(3)))
    return found


def is_hrum_app(pid: int, root: Path = ROOT) -> bool:
    """True when the process runs this repository's Streamlit app (agent/app/main.py, started from `root`)."""
    try:
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        cwd = Path(os.readlink(Path("/proc") / str(pid) / "cwd")).resolve()
    except OSError:
        return False
    return "streamlit" in cmd and "agent/app/main.py" in cmd and (cwd == root.resolve() or str(root) in cmd)


def hrum_app_ports(found: dict[int, list[int]] | None = None) -> list[int]:
    found = listeners() if found is None else found
    return sorted(port for port, pids in found.items() if any(is_hrum_app(pid) for pid in pids))


def resolve_app_port(explicit: int | None, env_port: str, detected: list[int]) -> tuple[int, str]:
    """(port, how it was chosen): --app-port, else APP_PORT in .env, else a detected HRUM app, else 8501."""
    if explicit:
        return explicit, "--app-port"
    if env_port.strip().isdigit():
        return int(env_port), "APP_PORT in .env"
    if detected:
        return detected[0], "detected HRUM process"
    return 8501, "default"


def check_services(cpu: bool) -> list[Result]:
    out = []
    if not VENV_PY.exists():
        return [Result(SKIP, "services", "needs .venv", "make setup")]
    if not cpu:
        code = """
import json, requests
from common.config import load_config
p = load_config("llm")["providers"]["qwen_local"]
base = p["base_url"].rstrip("/")
try:
    h = requests.get(base.rsplit("/v1", 1)[0] + "/health", timeout=3).status_code
    m = requests.get(base + "/models", headers={"Authorization": f"Bearer {p['api_key']}"}, timeout=3)
    root = m.json()["data"][0].get("root") if m.status_code == 200 else f"HTTP {m.status_code}"
except Exception as e:
    h, root = None, type(e).__name__
print(json.dumps({"health": h, "root": root}))
"""
        rc, res = _venv_python(code)
        info = json.loads(res.splitlines()[-1]) if rc == 0 else {"health": None, "root": res[-120:]}
        up = info["health"] == 200
        detail = f"up, serving {info['root']}" if up else "not running"
        if up and str(info["root"]).startswith("HTTP 401"):
            up, detail = False, "up, but QWEN_API_KEY does not match the running server"
        out.append(Result(OK if up else FAIL, "vLLM server (port 8000)", detail,
                          "make serve  (first start downloads the model and compiles kernels)"))
    env = read_env()
    optional = WARN if OPTS.demo else INFO        # optional services: a warning only when preparing a demo
    if env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"):
        containers = langfuse_containers()
        if containers is not None and containers[1]:
            up, total, bad = containers
            out.append(Result(OK if not bad else optional, "Langfuse containers", f"{up}/{total} up"
                              + (f"; not ok: {', '.join(bad)}" if bad else ""),
                              "cd ~/langfuse && docker compose up -d"))
        rc, res = _venv_python("from agent.observability import tracing; print(tracing.client().auth_check())")
        ok = rc == 0 and res.strip().endswith("True")
        out.append(Result(OK if ok else optional, "Langfuse API", "reachable, keys accepted" if ok else "not reachable",
                          "infra/langfuse/README.md (Troubleshooting)"))
    else:
        out.append(Result(INFO, "Langfuse", "keys not set: tracing off (optional)", "infra/langfuse/README.md"))

    found = listeners()
    detected = hrum_app_ports(found)
    port, how = resolve_app_port(OPTS.app_port, env.get("APP_PORT", ""), detected)
    name = f"HRUM app (port {port}, {how})"
    required = FAIL if OPTS.demo else INFO
    ok, body = http_ok(f"http://127.0.0.1:{port}/_stcore/health")
    healthy = ok and body == "ok"
    elsewhere = [p for p in detected if p != port]
    hint = (f"HRUM is running on port {elsewhere[0]}: make demo-check APP_PORT={elsewhere[0]}" if elsewhere
            else f"make agent APP_PORT={port}  (in its own terminal)")
    if healthy and port in detected:
        out.append(Result(OK, name, "Streamlit healthy, this repository's app"))
    elif healthy and found.get(port):          # we can see the owner, and it is not this repo's app
        out.append(Result(required, name, "a different Streamlit app is on this port (not HRUM from this repo)", hint))
    elif healthy:                              # owner not visible (another user or a container)
        out.append(Result(WARN, name, "a healthy Streamlit app, but its process is not visible to confirm it is HRUM",
                          hint))
    elif _port_open(port):
        out.append(Result(required, name, "port in use, but not a healthy Streamlit app (another program?)",
                          f"make agent APP_PORT={port + 1}  (then: make demo-check APP_PORT={port + 1})"))
    else:
        out.append(Result(required, name, "not running", hint))

    out.append(Result(OK if _port_open(MCP_PORT) else INFO, f"MCP server (port {MCP_PORT}, optional)",
                      "listening" if _port_open(MCP_PORT) else "not running", "make mcp"))

    if OPTS.demo and OPTS.smoke and not cpu:
        out.append(check_smoke())
    return out


def check_smoke() -> Result:
    """One real question end to end: model, tools, grounding (also warms the model up before a demo)."""
    rc, out = _run([str(VENV_PY), "-m", "agent.cli", "--smoke", "--json"], timeout=180)
    r = parse_cli_json(out)
    if rc != 0 or r is None:
        return Result(FAIL, "smoke question", (out.splitlines() or ["no output"])[-1][:160],
                      ".venv/bin/python -m agent.cli --smoke   (see the error)")
    ok = r.get("status") == "answered" and r.get("grounding", {}).get("ok")
    secs = r.get("usage", {}).get("seconds", "?")
    return Result(OK if ok else FAIL, "smoke question", f"{r.get('status')}, grounding ok={r.get('grounding', {}).get('ok')}, "
                                                        f"{secs} s", ".venv/bin/python -m agent.cli --smoke")

SECTIONS: list[tuple[str, Callable[..., list[Result]], bool]] = [
    ("Tools", check_tools, False), ("Configuration", check_env, False), ("App environment", check_venv, False),
    ("GPU", check_gpu, True), ("Model server software", check_vllm, True), ("Data", check_data, False),
    ("Model and index", check_artifacts, False), ("Running services", check_services, False),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", action="store_true", help="skip GPU / vLLM checks (laptop, CI)")
    ap.add_argument("--demo", action="store_true",
                    help="demo readiness: vLLM and the app must be up; runs one end-to-end question")
    ap.add_argument("--app-port", type=int, default=None,
                    help="port of the HRUM app (default: APP_PORT in .env, else a detected HRUM process, else 8501)")
    ap.add_argument("--no-smoke", action="store_true", help="with --demo: skip the end-to-end question")
    args = ap.parse_args()
    OPTS.cpu, OPTS.demo, OPTS.app_port, OPTS.smoke = args.cpu, args.demo, args.app_port, not args.no_smoke
    results: list[Result] = []
    for title, fn, gpu_only in SECTIONS:
        print(f"\n{title}")
        if gpu_only and args.cpu:
            res = [Result(SKIP, title, "--cpu")]
        else:
            res = fn(args.cpu) if fn in (check_tools, check_env, check_services) else fn()
        for r in res:
            print(f"  [{r.status:<4}] {r.name:<32} {r.detail}")
            if r.status in (FAIL, WARN) and r.fix:
                print(f"         -> {r.fix}")
        results += res
    fails = [r for r in results if r.status == FAIL]
    warns = [r for r in results if r.status == WARN]
    print(f"\n{len(fails)} fail, {len(warns)} warn.", end=" ")
    if fails:
        print(f"Next: {fails[0].fix}")
    else:
        print("Ready for the demo." if OPTS.demo else "Ready: make agent (APP_PORT=... if the port is taken)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
