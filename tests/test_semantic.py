import pytest

from common.semantic import SemanticError, compile_query, describe_catalog


def test_values_are_bound_as_parameters():
    q = compile_query(["headcount"], ["department"], {"department": "x'; drop table t; --"})
    assert "drop table" not in q.sql
    assert list(q.params.values()) == ["x'; drop table t; --"]


def test_min_group_size_is_enforced():
    q = compile_query(["avg_total_compensation"], ["job"])
    assert "having count(*) >= 5" in q.sql


def test_unknown_metric_and_dimension_rejected():
    with pytest.raises(SemanticError):
        compile_query(["salary_of_bob"])
    with pytest.raises(SemanticError):
        compile_query(["turnover_rate"], ["job"])  # job is not a snapshot dimension


def test_cross_model_metrics_rejected():
    with pytest.raises(SemanticError):
        compile_query(["headcount", "turnover_rate"])


def test_synthetic_caveat():
    q = compile_query(["turnover_rate"], ["gender"])
    assert any("SYNTHETIC" in c for c in q.caveats)
    assert describe_catalog()["metrics"]["turnover_rate"]["synthetic"] is True
    assert describe_catalog()["metrics"]["total_compensation"]["synthetic"] is False


def test_range_filter_and_limit_cap():
    q = compile_query(["total_compensation"], ["fiscal_year"], {"fiscal_year": {"gte": 2020, "lte": 2026}},
                      limit=10_000)
    assert q.params == {"p0": 2020, "p1": 2026}
    assert q.sql.endswith("limit 200")
