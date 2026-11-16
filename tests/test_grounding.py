import pytest

from agent.guardrails.grounding import extract_numbers, validate
from agent.tools.calc import safe_eval

LEDGER = {
    "E1": {"rows": [{"fiscal_year": 2026, "total_compensation": 7523952094.89, "headcount": 42925},
                    {"fiscal_year": 2025, "total_compensation": 6918855698.17}]},
    "E2": {"expression": "(7523952094.89/6918855698.17-1)*100", "result": 8.7456},
    "E3": {"rows": [{"turnover_rate": 0.1462}]},
}


@pytest.mark.parametrize("text,value", [
    ("$7.52B", 7.52e9), ("7,523,952,094", 7523952094), ("7,52 miliar", 7.52e9), ("Rp 1.234.567", 1234567),
    ("42.9K", 42900), ("14.6%", 14.6), ("3.1 million", 3.1e6),
])
def test_extract_numbers(text, value):
    assert extract_numbers(text)[0].value == pytest.approx(value)


def test_years_and_small_counts_are_trivial():
    nums = extract_numbers("In 2026 the top 3 departments")
    assert all(n.trivial for n in nums)
    assert extract_numbers("FY2026") == []


def test_grounded_answer_passes():
    ans = {"status": "answered", "answer": "FY2026 total compensation was $7.52B, up 8.7%; turnover 14.6%.",
           "claims": [{"text": "FY2026 total was $7.52B across 42,925 records", "evidence_ids": ["E1"]},
                      {"text": "Growth vs FY2025 was 8.7%", "evidence_ids": ["E2"]},
                      {"text": "Synthetic turnover was 14.6%", "evidence_ids": ["E3"]}]}
    v = validate(ans, LEDGER)
    assert v.ok, v.problems


def test_hallucinated_number_and_fake_citation_fail():
    ans = {"status": "answered", "answer": "Total was $8.1B",
           "claims": [{"text": "Total was $8.1B", "evidence_ids": ["E1"]},
                      {"text": "Turnover 12%", "evidence_ids": ["E9"]}]}
    v = validate(ans, LEDGER)
    assert not v.ok and v.claims_ok == 0 and len(v.problems) == 3


def test_number_cited_from_wrong_evidence_fails():
    ans = {"status": "answered", "answer": "", "claims": [{"text": "Growth 8.7%", "evidence_ids": ["E1"]}]}
    assert not validate(ans, LEDGER).ok


def test_refusal_without_numbers_passes():
    assert validate({"status": "refused", "answer": "I can't share an individual's pay.", "claims": []}, {}).ok


def test_answered_without_claims_fails():
    assert not validate({"status": "answered", "answer": "It went up.", "claims": []}, {}).ok


def test_safe_eval():
    assert safe_eval("(10 - 4) / 3 * 2 ** 2") == 8
    for bad in ("__import__('os')", "a + 1", "[1][0]"):
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(bad)


@pytest.mark.parametrize("text", ["Jumlahnya 42.930 baris.", "Jumlahnya 42,930 baris.", "Jumlahnya 42930 baris."])
def test_thousands_dot_or_comma_both_ground(text):
    ledger = {"E1": {"rows": [{"row_count": 42930}]}}
    ans = {"status": "answered", "answer": text, "claims": [{"text": text, "evidence_ids": ["E1"]}]}
    assert validate(ans, ledger).ok


def test_ambiguous_number_still_matches_decimal_reading():
    ledger = {"E1": {"rows": [{"ratio": 1.037}]}}
    ans = {"status": "answered", "answer": "Rasio 1.037", "claims": [{"text": "Rasio 1.037", "evidence_ids": ["E1"]}]}
    assert validate(ans, ledger).ok
    bad = {"status": "answered", "answer": "Rasio 1.037", "claims": [{"text": "Rasio 1.037", "evidence_ids": ["E1"]}]}
    assert not validate(bad, {"E1": {"rows": [{"x": 5.5}]}}).ok
