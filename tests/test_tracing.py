"""Tracing: span tree and PII masking, captured with an in-memory OpenTelemetry exporter (no Langfuse server)."""
import json
import logging

import pytest

from agent.core.loop import answer_question
from agent.observability import tracing
from agent.observability.tracing import mask, mask_text
from tests.conftest import requires_db

langfuse = pytest.importorskip("langfuse")


@pytest.fixture(scope="module")
def _langfuse():
    # The SDK keeps one client per public key, so one client + exporter serves the whole module
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse(public_key="pk-test", secret_key="sk-test", base_url="http://127.0.0.1:9",
                               span_exporter=exporter, tracer_provider=TracerProvider(), mask=mask)
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)   # scores cannot reach the fake host
    return client, exporter


@pytest.fixture
def spans(_langfuse):
    client, exporter = _langfuse
    exporter.clear()
    tracing.set_client(client)
    yield lambda: (client.flush(), exporter.get_finished_spans())[1]
    tracing.set_client(None)


@pytest.mark.parametrize("raw,masked", [
    ("Berapa gaji Stephen Lee di FY2026?", "Berapa gaji [NAME] di FY2026?"),
    ("email john.smith@example.com", "email [EMAIL]"),
    ("telp +62 812 3456 7890", "telp [PHONE]"),
    ("key 5d41402abc4b2a76b9719d911017c592", "key [KEY]"),
    ("Public Works, Transportation & Commerce", "Public Works, Transportation & Commerce"),
    ("Chief HR Officer of the City and County of San Francisco", "Chief HR Officer of the City and County of San Francisco"),
])
def test_mask_text(raw, masked):
    assert mask_text(raw) == masked


def test_vocabulary_has_no_example_names():
    # developer docs quote "Stephen Lee" as an example; they must not feed the allowlist
    from agent.observability.tracing import known_vocabulary

    assert not {"stephen", "john", "smith"} & known_vocabulary()


def test_mask_recurses_into_messages():
    msgs = [{"role": "user", "content": "Gaji Stephen Lee?"}]
    assert mask(msgs) == [{"role": "user", "content": "Gaji [NAME]?"}]
    assert mask_text("Hubungi John Smith di Police") == "Hubungi [NAME] di Police"


def test_tracing_off_is_a_noop():
    tracing.set_client(None)
    with tracing.observation("x", as_type="tool") as obs:
        obs.update(output=1)
        obs.score_trace(name="s", value=1)


@requires_db
def test_agent_run_produces_span_tree(scripted, spans):
    q_total = ("query_metric", {"metrics": ["total_compensation"], "filters": {"fiscal_year": 2026}})
    llm = scripted([[q_total], "draft",
                    {"status": "answered", "answer": "FY2026 total compensation was $7.52B.",
                     "claims": [{"text": "FY2026 total compensation was $7.52B", "evidence_ids": ["E1"]}]}])
    answer_question("Total kompensasi FY2026?", llm, trace={"tags": ["test"], "session_id": "s1"})
    by_name = {s.name: s for s in spans()}
    assert {"answer", "step_0", "step_1", "tool:query_metric", "finalize", "grounding"} <= set(by_name)
    attrs = by_name["answer"].attributes
    assert attrs["langfuse.observation.type"] == "agent"
    assert "test" in attrs["langfuse.trace.tags"] and attrs["langfuse.trace.name"] == "hrum.answer"
    assert by_name["step_0"].attributes["langfuse.observation.type"] == "generation"
    assert by_name["tool:query_metric"].parent.span_id == by_name["answer"].context.span_id


@requires_db
def test_names_never_reach_exported_spans(scripted, spans):
    llm = scripted(["Maaf, tidak bisa.", {"status": "refused", "answer": "Maaf, saya tidak bisa menampilkan gaji "
                                          "Stephen Lee.", "claims": []}])
    answer_question("Berapa gaji Stephen Lee di FY2026?", llm)
    exported = json.dumps([dict(s.attributes) for s in spans()], default=str)
    assert "Stephen" not in exported and "[NAME]" in exported


def test_unreachable_langfuse_turns_tracing_off_once(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")          # nothing listens here
    monkeypatch.setenv("HRUM_TRACING", "on")
    tracing.set_client(None)
    tracing._default_client.cache_clear()
    try:
        assert tracing.client() is None and not tracing.enabled()
        assert "not reachable" in (tracing.disabled_reason or "")
    finally:
        tracing._default_client.cache_clear()
        tracing.disabled_reason = None
