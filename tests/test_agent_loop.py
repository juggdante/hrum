"""End-to-end agent loop on the real DuckDB with a scripted (fake) LLM."""
from agent.core.loop import answer_question
from tests.conftest import requires_db

pytestmark = requires_db

Q_TOTAL = ("query_metric", {"metrics": ["total_compensation"], "dimensions": ["fiscal_year"],
                            "filters": {"fiscal_year": {"gte": 2025}}})


def test_happy_path_is_grounded_and_transparent(scripted):
    llm = scripted([
        [Q_TOTAL],
        [("calculate", {"expression": "(7523952094.89 / 6918855698.17 - 1) * 100", "label": "growth %"})],
        "Total compensation grew from 6.92B to 7.52B (+8.7%).",
        {"status": "answered", "answer": "FY2026 total compensation was $7.52B, up 8.7% from $6.92B in FY2025.",
         "claims": [{"text": "FY2026 total compensation was $7.52B", "evidence_ids": ["E1"]},
                    {"text": "FY2025 was $6.92B", "evidence_ids": ["E1"]},
                    {"text": "Growth was 8.7%", "evidence_ids": ["E2"]}]},
    ])
    r = answer_question("Berapa total kompensasi FY2026 dibanding FY2025?", llm)
    assert r.status == "answered" and r.repair_level == 0, r.grounding
    assert r.evidence[0]["sql"].startswith("select")
    assert r.evidence[0]["definitions"]["total_compensation"]
    assert any("Fiscal years" in c for c in r.caveats)
    assert r.usage["tool_calls"] == 2 and r.stop_reason == "done"


def test_hallucination_is_repaired_or_abstains(scripted):
    bad = {"status": "answered", "answer": "Total was $9.9B.",
           "claims": [{"text": "Total was $9.9B", "evidence_ids": ["E1"]}]}
    r = answer_question("Total kompensasi?", scripted([[Q_TOTAL], "draft", bad, bad]))
    assert r.status == "abstain" and r.repair_level == 3 and r.claims == []


def test_repair_keeps_only_supported_claims(scripted):
    mixed = {"status": "answered", "answer": "FY2026 was $7.52B and FY2024 was $5.0B.",
             "claims": [{"text": "FY2026 was $7.52B", "evidence_ids": ["E1"]},
                        {"text": "FY2024 was $5.0B", "evidence_ids": ["E1"]}]}
    r = answer_question("Total kompensasi?", scripted([[Q_TOTAL], "draft", mixed, mixed]))
    assert r.repair_level == 2 and len(r.claims) == 1 and "7.52B" in r.answer


def test_chro_cannot_use_run_sql(scripted):
    llm = scripted([[("run_sql", {"sql": "select count(*) from mart.fct_compensation"})], "draft",
                    {"status": "abstain", "answer": "Not available.", "claims": []}])
    r = answer_question("Run SQL please", llm, role="chro")
    assert any(e["kind"] == "tool_denied" for e in r.trace)
    assert all(e["tool"] != "run_sql" for e in r.evidence)   # only the auto doc search, never SQL


def test_analyst_sql_is_guarded(scripted):
    llm = scripted([[("run_sql", {"sql": "select total_compensation from mart.fct_compensation limit 1"})],
                    [("run_sql", {"sql": "select fiscal_year, count(*) as n from mart.fct_compensation "
                                         "where fiscal_year = 2026 group by 1"})],
                    "draft",
                    {"status": "answered", "answer": "42,925 paid records in FY2026.",
                     "claims": [{"text": "42,925 paid records in FY2026", "evidence_ids": ["E1"]}]}])
    r = answer_question("How many records in FY2026?", llm, role="analyst")
    assert "individual-level rows are blocked" in str(r.trace)
    assert "HAVING COUNT(*) >= 5" in r.evidence[0]["sql"]
    assert r.status == "answered", r.grounding


def test_calculate_rejects_numbers_not_in_evidence(scripted):
    llm = scripted([[("calculate", {"expression": "123456789 * 2"})], "draft",
                    {"status": "abstain", "answer": "No data.", "claims": []}])
    r = answer_question("x", llm)
    assert any(e["kind"] == "tool_failed" and "not in retrieved evidence" in e["error"] for e in r.trace)


def test_refusal_for_individual_lookup(scripted):
    llm = scripted(["Maaf, saya tidak bisa menampilkan gaji individu.",
                    {"status": "refused", "answer": "Maaf, saya tidak bisa menampilkan gaji individu.",
                     "claims": []}])
    r = answer_question("Berapa gaji Stephen Lee?", llm)
    assert r.status == "refused" and not any(e["cited"] for e in r.evidence)


def test_small_model_argument_mistakes_are_normalized(scripted):
    llm = scripted([[("query_metric", {"metrics": "Turnover_Rate", "dimensions": '["gender"]', "limit": "5"})],
                    "draft", {"status": "abstain", "answer": "x", "claims": []}])
    r = answer_question("turnover by gender", llm)
    assert r.evidence and r.evidence[0]["arguments"]["metrics"] == ["turnover_rate"]
    assert any("SYNTHETIC" in c for c in r.evidence and answer_caveats(r))


def answer_caveats(r):
    from common.semantic import compile_query

    return compile_query(["turnover_rate"], ["gender"]).caveats


def test_attrition_insight_is_aggregated_and_labelled(scripted):
    llm = scripted([[("attrition_insight", {"segment_type": "organization_group", "top_n": 3})], "draft",
                    {"status": "abstain", "answer": "x", "claims": []}])
    r = answer_question("Siapa yang berisiko resign?", llm)
    ev = r.evidence[0]
    assert len(ev["rows"]) == 3 and all(row["employee_count"] >= 5 for row in ev["rows"])
    assert "employee_key" not in str(ev)


def test_duplicate_citations_are_collapsed():
    from agent.core.loop import dedupe_citations

    ans = dedupe_citations({"claims": [{"text": "x", "evidence_ids": ["E1", "E1", "E2"]}]})
    assert ans["claims"][0]["evidence_ids"] == ["E1", "E2"]


def test_no_tool_call_triggers_doc_retrieval(scripted):
    from agent.rag.index import index_dir

    if not (index_dir() / "chunks.parquet").exists():
        import pytest
        pytest.skip("run `make index`")
    llm = scripted([
        "Tidak ada bukti.",   # model skipped tools
        {"status": "answered", "answer": "Tahun fiskal dimulai 1 Juli.",
         "claims": [{"text": "Tahun fiskal dimulai 1 Juli", "evidence_ids": ["E1"]}]},
    ])
    r = answer_question("Tahun fiskal di data ini mulai bulan apa?", llm)
    assert r.evidence[0]["tool"] == "search_docs" and r.status == "answered"
    assert any(e["kind"] == "auto_retrieve" for e in r.trace)
    assert "July 1" in llm.calls[-1]["messages"][0]["content"]   # the FAQ passage reached finalisation


def test_no_auto_retrieval_when_a_tool_ran(scripted):
    llm = scripted([[Q_TOTAL], "draft",
                    {"status": "answered", "answer": "FY2026 total compensation was $7.52B.",
                     "claims": [{"text": "FY2026 total compensation was $7.52B", "evidence_ids": ["E1"]}]}])
    r = answer_question("Total kompensasi FY2026?", llm)
    assert [e["tool"] for e in r.evidence] == ["query_metric"]
