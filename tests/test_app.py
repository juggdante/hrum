"""Streamlit app: every page renders, chat answers with a chart, question bank and SQL Explorer work.

Fake LLM + real DuckDB. AppTest runs the real multipage entry point (agent/app/main.py).
"""
import json

import pytest

from common.config import ROOT
from tests.conftest import requires_db

pytestmark = requires_db
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(ROOT / "agent" / "app" / "main.py")


class _FakeLLM:
    """First step calls query_metric (total compensation by fiscal year), then answers grounded."""
    name = model = "fake"

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None, json_schema=None, max_tokens=800):
        from agent.llm.base import LLMResponse, ToolCall

        self.n += 1
        if json_schema:
            return LLMResponse(text=json.dumps({
                "status": "answered", "answer": "FY2026 total compensation was $7.52B.",
                "claims": [{"text": "FY2026 total compensation was $7.52B", "evidence_ids": ["E1"]}]}))
        if self.n == 1:
            return LLMResponse(tool_calls=[ToolCall(id="c1", name="query_metric", arguments=json.dumps(
                {"metrics": ["total_compensation"], "dimensions": ["fiscal_year"],
                 "filters": {"fiscal_year": {"gte": 2022}}}))])
        return LLMResponse(text="draft")


@pytest.fixture
def fake_llm(monkeypatch):
    from agent.llm import base

    monkeypatch.setattr(base, "get_client", lambda provider=None: _FakeLLM())


def _app(role="chro"):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["role"] = role
    return at.run()


def test_chat_answer_has_chart_and_evidence(fake_llm):
    at = _app()
    assert not at.exception
    assert "anthropic" not in at.sidebar.selectbox[0].options         # no unimplemented providers
    at.chat_input[0].set_value("Total kompensasi FY2022-2026?").run()
    assert not at.exception
    assert "7.52B" in " ".join(m.value for m in at.markdown)
    assert at.get("plotly_chart"), "a line chart of the cited year series is expected"
    assert any("select" in c.value for c in at.code)                   # the SQL is shown


def test_chat_shows_error_when_model_is_down(monkeypatch):
    from agent.llm import base

    class Down(_FakeLLM):
        def chat(self, *a, **k):
            raise ConnectionError("vLLM down")

    monkeypatch.setattr(base, "get_client", lambda provider=None: Down())
    at = _app()
    at.chat_input[0].set_value("Total kompensasi FY2026?").run()
    assert not at.exception and "Model tidak bisa dihubungi" in at.error[0].value


def test_dataset_page_renders_live_counts():
    at = _app().switch_page("views/dataset.py").run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Tahun fiskal"] == "FY2013-FY2026"
    assert at.get("plotly_chart") and len(at.expander) > 5


def test_question_bank_sends_question_to_chat(fake_llm):
    at = _app().switch_page("views/questions.py").run()
    assert not at.exception
    at.button(key="ask-analyst_01").click().run()
    assert not at.exception
    assert at.session_state["role"] == "analyst"                       # analyst question switches the role
    assert any("Pakai SQL" in m.value for m in at.markdown)            # question was asked on the chat page


def test_sql_explorer_not_registered_for_chro():
    with pytest.raises(ValueError, match="Could not find a navigation page"):
        _app("chro").switch_page("views/sql_explorer.py")


def test_sql_explorer_runs_row_level_and_rejects_bad_sql():
    at = _app("analyst").switch_page("views/sql_explorer.py").run()
    assert not at.exception
    at.text_area(key="sql_text").set_value("select * from mart.fct_compensation where fiscal_year = 2026 limit 5")
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("EXCLUDE" in c.value for c in at.code)                 # person keys removed from SELECT *
    assert "employee_key" not in at.dataframe[-1].value.columns
    at.text_area(key="sql_text").set_value("select * from raw.sf_employee_compensation")
    at.button[0].click().run()
    assert at.error and "ditolak" in at.error[0].value
