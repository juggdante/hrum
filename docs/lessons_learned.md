# Lessons learned: dos, don'ts, tips and gotchas

A living document. Add an entry whenever something costs more than ten minutes to figure out.
Each entry says what happened, why, and what to do.

_Last updated: 2026-11-12 (Langfuse tracing)._

## Setup and environment

**Do**
- **On a shared account, check before you stop or start vLLM.** `make serve` and `vllm_server stop` use
  `pkill -f 'vllm serve'`, which stops every vLLM of the user account, including a colleague's. Look first with
  `ps -eo pid,user,lstart,cmd | grep "[v]llm serve"` and `nvidia-smi`. A PID-based stop was considered and
  deliberately not built (risk of a new process-killing change); the check is a manual habit for now.
- Install vLLM in its own venv at `~/.venvs/vllm`. vLLM pins its own torch build, which would conflict with the
  app venv. The launcher finds that location automatically; for another location set `VLLM_BIN` in `.env`
  (not with `export`, see below).
- Run long jobs on the GPU PC inside `tmux`, so they survive a dropped SSH connection:
  `tmux new -s hrum` → run `make serve && make eval` → detach with `Ctrl-b d` → after reconnecting,
  `tmux attach -t hrum`. Without tmux, a dropped connection kills `make eval` halfway (the vLLM server keeps
  running, because the launcher starts it in its own session; check with `vllm_server status`).
- **tmux stops accepting typing:** usually you are in copy mode after scrolling (a `[0/123]` counter appears at
  the top right). Press `q` or `Esc`. If the screen froze after `Ctrl-s` (terminal flow control), press `Ctrl-q`.
- **Copying output out of tmux:** selecting with the mouse puts tmux in copy mode, which looks like a freeze.
  Write output to a file instead (`make failures > logs/failures.txt`; the gate writes `evals/reports/gate.md`)
  and copy from the VS Code editor, or hold `Shift` while selecting.
- **tmux inside the VS Code terminal:** VS Code grabs `Ctrl-b` (toggle sidebar), so the tmux prefix never reaches
  tmux and `Ctrl-b d` does nothing. Either set `"terminal.integrated.sendKeybindingsToShell": true` in VS Code
  settings, or run tmux from a plain SSH terminal. A session left attached by a dropped connection can also
  look frozen: from a new terminal run `tmux attach -d -t hrum` (`-d` detaches the stale client).
- Run data, dbt and tests on any CPU machine. Only `make serve` and `make eval` need the GPU.
- Generate your own secrets. `QWEN_API_KEY` is a password you invent for your own vLLM server (it is passed as
  `--api-key`). `HRUM_JWT_SECRET` signs MCP tokens. Neither comes from an outside service:
  `python3 -c "import secrets; print(secrets.token_urlsafe(24)); print(secrets.token_hex(32))"`.
- Fill only what you need in `.env`: `QWEN_API_KEY` to start; `HRUM_JWT_SECRET` before `make mcp`; `HF_TOKEN`
  is optional; leave cloud keys and Langfuse empty until they are used.

**Gotchas**
- **`make serve` says the vllm file is missing:** `export VLLM_BIN` only says where vLLM is; it does not
  install it. The launcher now prints the install command.
- **`vLLM not found at 'vllm'` after reconnecting over SSH:** `export` lives only as long as the shell. A new
  SSH login starts without it. Fixed for good: the launcher checks `VLLM_BIN` (shell or `.env`), then
  `~/.venvs/vllm/bin/vllm`, then `PATH`. Rule of thumb: settings that must survive go in `.env` or
  `~/.bashrc`, never only in an interactive `export`.
- **`nvcc fatal: Unknown option '--compress-mode=size'`:** FlashInfer JIT-compiles its sampler with the system
  `nvcc`, and an older CUDA toolkit does not know that flag. The GPU driver can be new (CUDA 13.2) while
  `/usr/bin/nvcc` is old. The launcher sets `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch sampler, nothing to
  compile). After a failed build, delete `~/.cache/flashinfer`.
- **"Engine core initialization failed" is only a summary.** The real error is in the `EngineCore` lines of
  `logs/vllm.log`. Do not guess "out of memory" from the startup phase: check `nvidia-smi` and grep the log
  first. In our case the GPU was empty and the cause was the compiler.
- **VS Code offers to inject `.env` into terminals** (`python.terminal.useEnvFile`). Leave it off. HRUM loads
  `.env` itself on every run, and python-dotenv does not override variables that already exist, so an injected
  copy goes stale after you edit `.env` (until the terminal is reopened). Injection is also the likely way a
  relative `HRUM_DB_PATH` ended up in the shell and broke dbt.
- **The first start of a new model is slow** (download plus kernel compilation, about 15 minutes for 4B).
  Later starts take about 30 seconds.
- **A relative `HRUM_DB_PATH` breaks dbt.** If `.env` is exported into the shell, `data/hrum.duckdb` resolves
  against `warehouse/` because dbt runs there. The Makefile now always passes dbt an absolute path.

- **Pin dependencies with `uv.lock`.** Unpinned installs already broke us twice (pandas 3 string dtype, a
  deprecated Streamlit argument). `make setup` is `uv sync --locked`; `make lock-upgrade` upgrades on purpose, then
  run `make test` and the evals before committing the new lock.
- **When to change the lock.** Upgrade on purpose, never routinely before a demo or a submission (the eval results
  hold for the versions in the lock).

  | Situation | Command |
  |---|---|
  | Add a dependency | add it to `pyproject.toml`, then `uv lock` (other packages stay put) |
  | Security fix or needed bug fix in one package | `uv lock --upgrade-package <name>` |
  | Planned maintenance (e.g. quarterly), new Python version | `make lock-upgrade` (everything) |

  After any lock change: `make test` on the laptop, `make eval` + `make gate` on the GPU PC, then commit `uv.lock`.
- **`requires-python` shapes the lock.** With `>=3.10`, uv resolved a second, older set for Python 3.10 (Langfuse 4,
  pandas 2) that the code was never tested with. It is now `>=3.11`, the versions actually used.
- **Torch from the PyPI default is the CUDA build** (GBs of `nvidia-*` wheels). The app venv only needs CPU torch
  (the GPU belongs to vLLM's own venv), so `pyproject.toml` pins torch to the PyTorch CPU index. `[tool.uv.sources]`
  applies only to direct dependencies, so torch is listed explicitly in the `rag` extra.

## Data pipeline

**Do**
- Run `make build-data` on a fresh machine. It runs ingest → `make synth` (SF Comp staging, then synthetic HRIS)
  → `dbt build` in the right order.
- Set `SF_COMP_CSV` to a CSV you already downloaded, to skip the 400 MB download.

**Gotchas**
- **Chicken-and-egg in staging.** `stg_synthetic__hris` needs `raw.synthetic_hris`, which `make synth` creates.
  `make synth` therefore builds only `stg_sf_comp__compensation` first. It worked on the laptop only because
  the table already existed from an earlier run. Always test the build on a fresh database.
- **A fixed seed is not enough for reproducibility.** DuckDB returns rows in a nondeterministic order (parallel
  scan), so the same random numbers landed on different employees on each run. The laptop and the GPU PC
  disagreed (Elections risk 0.41 vs 0.37). Any query that feeds a seeded RNG or a train/test split needs an
  `ORDER BY` on a unique key, plus a tie-breaker in window functions.
- **"Headcount" counts pay records, not people.** A person with two jobs counts twice. The `headcount` metric
  also filters `is_paid` (42,925 in FY2026); a raw `count(*)` gives 42,930.
- Two FY2017 rows have no department. They are kept as `UNK` so city totals reconcile.

## Agent behaviour (Qwen3.5-4B)

**What works**
- The tool-calling format matches the `qwen3_coder` parser with thinking turned off. No format errors so far.
- The model refuses questions about individuals and destructive requests before any tool runs. The SQL guard
  is a second line of defence that the eval attacks must reach separately.
- Grounding with one repair round: 7 of 39 answers were fixed by the repair step and then passed.

**Gotchas**
- **The prompt decides when a tool is used.** "Use `run_sql` only when no metric fits" made the model abstain
  when the user explicitly asked for SQL. Rules should name the exceptions.
- **Small models fan out.** Asked for "the riskiest segments", the model called the attrition tool once per
  segment type (6 calls, 16k tokens). Tool design fixed it better than prompting: the default mode returns an
  overview in one call.
- **Duplicate citations** (`["E1", "E1"]`) happen. They are collapsed after finalisation.
- **Number style drifts toward the user's language.** In Indonesian answers the model writes "9,9%" despite an
  "English style" rule. The rule now gives an explicit wrong example.
- **Indonesian number style breaks naive parsing.** "42.930" is 42,930 in Indonesian, but a parser that assumes
  English reads 42.93. A correct answer then failed grounding and became an abstain. One separator followed by
  exactly three digits is ambiguous; the grounding check now accepts both readings.
- **Models sometimes skip tools** on "why/how" questions and then abstain for lack of evidence. A deterministic
  safety net (search the docs with the question when no tool ran) is more reliable than another prompt rule.
- **"refused" vs "abstain" needs definitions.** The 9B model "refused" a question that was allowed but had no
  data. The finalize prompt now says: refused only for disallowed requests; abstain when the data cannot answer.
- **Glued words** ("karyawantermasuk") appear in some answers under JSON-schema decoding. Not fixed yet; the
  eval measures it.
- **Caveats follow the table, not the columns.** Any answer from the snapshot table gets the "synthetic" caveat,
  even when only real columns were used.

## Evaluation

**Do**
- Compute gold answers with SQL (`gold_sql`), never type them in. They stay correct after a data rebuild.
- Make every gold query mirror the metric definition exactly. **Trap:** `avg(flag::int)` ignores NULLs, so the
  voluntary turnover gold came out at 70% (share of leavers) instead of 10.2% (share of all employees). Use
  `avg(case when flag then 1.0 else 0.0 end)`, as the metric does.
- Check the harness with a fake LLM first (always-refuse should fail the gate). It is cheap and catches wiring bugs.
- Compare models with the **same prompt version and question set**. Reports record both, and the gate flags a
  mismatch. Re-run the baseline whenever the prompt changes.
- Use the paired McNemar test, not two separate accuracies. With 39 questions, only the questions where the
  models disagree carry information.

- Check what the user actually sees. A required phrase can be satisfied by a system-attached caveat; grading
  only the model's sentence failed a correct, compliant answer (risk_01).
- Keep grading separate from generation: `make regrade` re-scores saved runs with the current checks, so a
  grader fix does not need another GPU run.

**Don't**
- Don't trust a pass rate without its interval: 37/39 has a 95% CI of 83-99%.
- Don't grade style (decimal commas, separators) as pass/fail yet. Report it, fix it, then promote it to a
  check once it is stable.
- Don't let an "attack" pass just because the status is right. Also check that no forbidden number (for
  example the pay of a single-person job) or hashed key appears in the text.

**Retrieval gotcha**
- **Hybrid search can be worse than dense alone.** With an Indonesian question and English docs, BM25 had no real
  overlap and ranked chunks by noise words ("data", score ~0.5). In RRF those noise ranks outvoted the dense
  retriever, which had the right chunk at rank 1. BM25 now votes only for chunks with score ≥ 1.0
  (`min_lexical_score` in `configs/rag.yaml`). Always test retrieval with questions in the users' language.

- **A glob in the RAG sources indexes everything you write later.** `docs/*.md` silently pulled the progress
  report, lessons learned (with example names) and setup guide into the agent's knowledge. Sources are now an
  explicit list that matches what the evals ran against.

**Tips**
- An eval with the model server down used to "run" every question as an error and report 0/N under the label
  `qwen`. `make eval` now checks the server first (reachable, key accepted) and stops with the fix.
- `make failures` prints the failed questions of the latest run per model, with tools and grounding problems.
  Run files keep the full trace, so a failure can be explained without re-running.
- `make eval ARGS="--only cost_01,atk_01"` for a quick check; `--repeat 3` for pass^k.
- Answers and traces are in `evals/reports/runs/*.jsonl` (gitignored). Summaries are in `evals/reports/<model>.json`.

## Observability (Langfuse)

**Do**
- Read the installed SDK's signatures before writing code against it (`inspect.signature`). The SDK here was
  Langfuse 5.2: traces are built from OpenTelemetry observations (`start_as_current_observation(as_type=...)`,
  `propagate_attributes` for trace name/tags/session); there is no `update_trace`, and a trace's input/output
  come from its root observation.
- Test tracing without a server: pass an OpenTelemetry `InMemorySpanExporter` as `span_exporter` and assert on
  the exported span attributes (for example that a name never appears).
- Mask with the SDK `mask` hook so nothing unmasked leaves the process, and make it recurse: LLM inputs are lists
  of message dicts, not strings.

- Self-hosting: put the compose secrets in `~/langfuse/.env` (every `CHANGEME` is `${VAR:-default}`) and
  generate them straight into the file, so they are never printed or pasted into a chat. Bind every published
  port to `127.0.0.1`; the official file exposes the UI, MinIO and the AI gateway on all interfaces.

**Gotchas**
- **One client per public key.** The SDK reuses its tracer for the same public key, so a second test client
  silently exported into the first test's exporter. Use one client per test module and clear the exporter.
- **Masking names by regex over-masks.** Mask only the unknown words inside a capitalised phrase ("Gaji Stephen
  Lee" → "Gaji [NAME]"), and allowlist HRUM's own vocabulary so departments and job titles survive.
- **An allowlist built from docs can leak.** The name rule first learned its vocabulary from all of `docs/`, and
  this very file quotes "Stephen Lee" as an example, so the name stopped being masked. The vocabulary now comes
  only from sources that never hold names (dimensions, metric catalogue, knowledge base, prompts), and a test
  asserts that example names are not in it.
- **Traces reveal cold starts.** The first docs question looked slow (14.8 s); the per-stage split showed ~10 s
  was loading the embedding model. Load heavy models at start-up, and measure latency after warm-up.
- **Streamlit binds to all interfaces by default** (port 8501), like the Langfuse compose file. `make agent` now
  passes `--server.address 127.0.0.1`; open it from the laptop through VS Code port forwarding.
- **A health check proves a service is up, not that it is yours.** `make doctor` once reported "HRUM app healthy"
  on 8501, where a colleague's Streamlit app was running. It now identifies HRUM's own process (`ss` pid →
  `/proc/<pid>/cmdline` and `cwd`) and uses that port unless `APP_PORT` is given.
- **Port taken on a shared machine** ("Port 8501 is not available"): check the owner with `ss -ltnp | grep 8501`;
  never stop someone else's process. `make agent APP_PORT=8502` picks another port.
- **Streamlit sends usage statistics by default.** Off for an on-prem app: `--browser.gatherUsageStats false`.
- **Test the Streamlit app without a browser** with `streamlit.testing.v1.AppTest` (render, type a question,
  assert on markdown/code/error elements) and a fake LLM; pass an absolute script path.
- **Streamlit multipage:** keep pages in a folder not named `pages/` (that name triggers the legacy auto-multipage
  mode). A widget's session state can only be set before the widget is created, so a page that wants to change
  the sidebar role sets `role_request`, which the entry script applies before drawing the radio.
- **Do not disable a button until a text area has content.** The text area commits on blur, so the first click
  after typing lands on a still-disabled button and is lost. Keep it enabled and validate on click.
- **pandas 3 text columns have the `str` dtype**, not `object`; type checks must accept both.
- **`use_container_width` is deprecated** in recent Streamlit; use `width="stretch"`.
- **A forwarded port can go stale** after the SSH connection drops and VS Code reconnects: the page just keeps
  loading. First check the service on the PC (`curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000`);
  if it answers, Stop Forwarding the port in the VS Code Ports tab and forward it again. Also read the
  "Forwarded Address" column: VS Code may map 3000 to another local port if 3000 is busy on the laptop.
- **Finding a question's trace in the Langfuse UI:** the Tracing table lists every observation. Filter
  "Is Root Observation = True" (one row per question), or open Sessions → the eval run.
- **Tests must not reach the real Langfuse.** The suite reads the machine's `.env`, so on the GPU PC it tried to
  export traces of fake-LLM runs. `tests/conftest.py` sets `HRUM_TRACING=off` (HRUM's own switch:
  `LANGFUSE_TRACING_ENABLED` is also read by the SDK and would disable the in-memory client the tracing tests use).
- **Keys set but Langfuse down** used to flood the output with export retries. The agent now probes
  `/api/public/health` once at start and turns tracing off with one warning; answers never depend on tracing.
- **Flush before a short-lived process exits** (CLI, eval). The SDK sends in the background.

## Git and repository hygiene

**Do**
- Keep course materials and personal notes in `private/`, outside the repo. Only `hrum/` is a git repository.
- **Test the setup from a second clone, not your working copy.** `git clone <url> hrum-fresh` (run from the parent
  folder) creates `hrum-fresh/` directly. A plain `git clone <url>` creates a folder named after the repo (`hrum`)
  and is refused if that folder exists; `mkdir hrum-fresh && cd hrum-fresh && git clone <url>` nests it as
  `hrum-fresh/hrum`. `clone` makes a new copy; `pull` only updates an existing one. A second clone shares the
  machine's caches (Hugging Face models, uv, the vLLM venv, Docker), so it tests the repo and the guide, not a
  bare machine; stop the old vLLM first so `make serve` runs with the new clone's `QWEN_API_KEY`.
- **Check what you are about to commit** (`git status --short`). An export step once dropped a copy of the root
  `Makefile` into `docs/deck/`; it was caught as an untracked file before the commit.
- Push from the laptop and pull on the GPU PC. The office Git server is reachable only on the office
  network or VPN. If a push fails with "No route to host", the commit is still safe locally; push later.
- Commit summaries such as eval reports and the model card. Never commit `.env`, raw data, DuckDB files or
  the run logs in `evals/reports/runs/`.

**Gotchas**
- **Shared GPU machine, someone else's git identity.** The global `user.email` on a shared PC belongs to whoever
  set it first, so commits made there carry their name. Set a repo-local identity (`git config user.name ...`,
  `git config user.email ...` inside the repo, no `--global`) and check with `git log -1 --format='%an <%ae>'`.
  An unpushed commit can be fixed with `git commit --amend --reset-author`.
- **Push rejected ("fetch first"):** the laptop pushed while the PC had local commits. Run `git pull --rebase`,
  then push. Pull before running anything that depends on new code (`make regrade` did not exist yet on the PC).

**Before publishing to a personal GitHub**
- Check the policy first: the code was developed on office hardware.
- Rewrite commit authors (`git filter-repo --mailmap`) and search the history for internal host names,
  e-mails and paths.
- Publish as a private repository first, review it, then make it public.

## Working agreements
- Code, comments, commits and docs are in English; discussion is in Indonesian.
- Every fix found on the GPU PC is reproduced on the laptop (fresh DB or fake LLM) before pushing.
- Update `docs/progress_report.md` after each milestone and this file after each new gotcha.
