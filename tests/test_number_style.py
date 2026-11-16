import pytest

from agent.guardrails.grounding import extract_numbers
from agent.guardrails.number_style import normalize_answer, normalize_numbers


@pytest.mark.parametrize("raw,expected", [
    ("Turnover 14,56% (0,1456).", "Turnover 14.56% (0.1456)."),
    ("naik 9,9% dari FY2022", "naik 9.9% dari FY2022"),
    ("Total 7.523.952.094,89 USD", "Total 7,523,952,094.89 USD"),
    ("sebesar 5,5 juta", "sebesar 5.5 juta"),
    ("42.930 baris", "42.930 baris"),          # ambiguous: untouched
    ("rasio 1,037", "rasio 1,037"),            # ambiguous: untouched
    ("1,2,3 dan FY2022,2023", "1,2,3 dan FY2022,2023"),
    ("$7.52B dan 1,234,567", "$7.52B dan 1,234,567"),
])
def test_normalize_numbers(raw, expected):
    assert normalize_numbers(raw) == expected


@pytest.mark.parametrize("raw", ["14,56%", "0,1456", "7.523.952.094,89", "5,5 juta", "1.234.567"])
def test_normalization_never_changes_the_value(raw):
    before = [n.value for n in extract_numbers(raw)]
    after = [n.value for n in extract_numbers(normalize_numbers(raw))]
    assert before == pytest.approx(after)


def test_normalize_answer_covers_claims():
    a = normalize_answer({"answer": "Turnover 14,56%.", "claims": [{"text": "14,56%", "evidence_ids": ["E1"]}]})
    assert a["answer"] == "Turnover 14.56%." and a["claims"][0]["text"] == "14.56%"
