export MLFLOW_DISABLE_AGENT_HINT=1
.PHONY: index setup ingest synth dbt-run dbt-test build-data train serve agent mcp eval regrade failures gate trace-check doctor demo-check test lint
PY := .venv/bin/python
# `make setup QUIET=` shows uv's full install output
QUIET ?= -q
APP_PORT ?= 8501
# dbt runs inside warehouse/, so hand it an absolute DB path (a relative HRUM_DB_PATH is relative to the repo root)
DB_PATH := $(abspath $(or $(HRUM_DB_PATH),data/hrum.duckdb))
DBT := cd warehouse && DBT_PROFILES_DIR=. HRUM_DB_PATH=$(DB_PATH) ../.venv/bin/dbt

setup:        ## create venv if missing + install/update deps (safe to re-run) (CPU side; vLLM has its own venv, see infra/local_server)
	test -d .venv || uv venv $(QUIET) .venv
	uv pip install $(QUIET) --python .venv/bin/python -e ".[dev,ml,rag,agent]"
ingest:       ## SF Employee Compensation -> DuckDB raw (set SF_COMP_CSV to reuse a local file)
	$(PY) -m ingestion.sources.sf_comp
synth:        ## synthetic HRIS layer -> DuckDB raw (builds staging first; needs `make ingest`)
	$(DBT) run --select stg_sf_comp__compensation
	$(PY) -m ingestion.synthetic.generate
dbt-run:
	$(DBT) run
dbt-test:
	$(DBT) test
build-data: ingest synth  ## full rebuild: raw -> staging -> synthetic -> marts + tests
	$(DBT) build
train:       ## train attrition model + publish aggregated risk to mart
	$(PY) -m ml.attrition.train
	$(PY) -m ml.attrition.score
index:       ## build documentation index for RAG
	$(PY) -m agent.rag.index
serve:        ## start vLLM (profile from configs/llm.yaml, default qwen35_4b)
	$(PY) -m infra.local_server.vllm_server start $(or $(PROFILE),qwen35_4b)
agent:        ## Streamlit chat on 127.0.0.1:$(APP_PORT) (APP_PORT=8502 if 8501 is taken; forward it in VS Code)
	.venv/bin/streamlit run agent/app/main.py --server.address 127.0.0.1 --server.port $(APP_PORT) \
		--server.headless true --browser.gatherUsageStats false
mcp:          ## MCP server, Streamable HTTP with JWT (use TRANSPORT=stdio for local clients)
	$(PY) -m agent.mcp.server --transport $(or $(TRANSPORT),http)
eval:         ## golden questions vs the served model -> evals/reports/<model>.json (ARGS="--only cost_01")
	$(PY) -m evals.run_eval $(ARGS)
regrade:      ## re-grade saved runs with the current checks, no GPU (ARGS="qwen3.5-4b qwen3.5-9b")
	$(PY) -m evals.run_eval --regrade $(or $(ARGS),qwen3.5-4b qwen3.5-9b)
failures:     ## failed questions of the latest eval run per model (answers, tools, grounding)
	$(PY) -m evals.show $(ARGS)
gate:         ## release gate: CANDIDATE=evals/reports/x.json [BASELINE=evals/reports/y.json]
	$(PY) -m evals.gate --candidate $(or $(CANDIDATE),evals/reports/qwen3.5-9b.json) \
		$(if $(or $(BASELINE),$(CANDIDATE)),$(if $(BASELINE),--baseline $(BASELINE)),--baseline evals/reports/qwen3.5-4b.json)
doctor:       ## check prerequisites + setup state, print the next step (ARGS=--cpu on a laptop)
	$(if $(wildcard .venv/bin/python),.venv/bin/python,python3) -m infra.doctor $(ARGS)
demo-check:   ## before a demo: vLLM + app must be up, optional services, one end-to-end question
	$(PY) -m infra.doctor --demo $(if $(filter command line environment,$(origin APP_PORT)),--app-port $(APP_PORT)) $(ARGS)
trace-check:  ## Langfuse connection + one masked test trace (infra/langfuse/README.md)
	$(PY) -m agent.observability.check
test:
	.venv/bin/pytest -q
lint:
	.venv/bin/ruff check .
