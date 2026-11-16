"""Chart choice per evidence shape, on real semantic-layer results."""
import duckdb
import pandas as pd
import pytest

from agent.app.charts import (
    MAX_BARS,
    chart_for_table,
    charts_for_attrition,
    charts_for_evidence,
    charts_for_query_metric,
    format_value,
)
from common.config import db_path
from common.semantic import catalog, compile_query, run_query
from tests.conftest import requires_db


def _evidence(metrics, dimensions=None, filters=None, limit=50, eid="E1"):
    con = duckdb.connect(str(db_path()), read_only=True)
    q = compile_query(metrics, dimensions, filters, limit=limit)
    res = run_query(con, q)
    return {"id": eid, "tool": "query_metric", "cited": True, "rows": res["rows"],
            "arguments": {"metrics": metrics, "dimensions": dimensions or [], "filters": filters or {}}}


@pytest.mark.parametrize("v,fmt,out", [
    (7_523_952_094.89, "usd", "$7.52B"), (163_425_078, "usd", "$163.43M"), (0.1456, "percent", "14.6%"),
    (0.31664, "ratio", "0.317"), (42925, "integer", "42,925"), (None, "usd", "n/a"),
])
def test_format_value(v, fmt, out):
    assert format_value(v, fmt) == out


@requires_db
def test_scalar_becomes_kpi():
    [spec] = charts_for_query_metric(_evidence(["total_compensation", "headcount"],
                                               filters={"fiscal_year": 2026}), catalog())
    assert spec.kind == "kpi" and ("Total compensation", "$7.52B") in spec.kpis
    assert ("Headcount", "42,925") in spec.kpis


@requires_db
def test_year_series_is_a_line_and_units_never_share_an_axis():
    specs = charts_for_query_metric(_evidence(["total_compensation", "headcount"], ["fiscal_year"],
                                              {"fiscal_year": {"gte": 2022}}), catalog())
    assert len(specs) == 2                                  # USD and count: two charts, no dual axis
    fig = specs[0].figure
    assert fig.data[0].type == "scatter" and list(fig.data[0].x) == [2022, 2023, 2024, 2025, 2026]
    assert fig.layout.yaxis.title.text == "USD billions"


@requires_db
def test_category_is_a_sorted_capped_bar():
    [spec] = charts_for_query_metric(_evidence(["total_overtime"], ["department"], {"fiscal_year": 2026},
                                               limit=200), catalog())
    bar = spec.figure.data[0]
    assert bar.type == "bar" and bar.orientation == "h" and len(bar.y) == MAX_BARS
    assert bar.y[-1] == "Police"                              # largest at the top
    assert "top 15" in spec.caption


@requires_db
def test_year_by_category_is_capped_lines():
    [spec] = charts_for_query_metric(_evidence(["total_compensation"], ["fiscal_year", "organization_group"],
                                               {"fiscal_year": {"gte": 2024}}, limit=200), catalog())
    assert {t.type for t in spec.figure.data} == {"scatter"} and 2 <= len(spec.figure.data) <= 8


@requires_db
def test_two_categories_is_a_heatmap_and_synthetic_is_labelled():
    [spec] = charts_for_query_metric(_evidence(["turnover_rate"], ["organization_group", "level"], limit=200),
                                     catalog())
    assert spec.figure.data[0].type == "heatmap" and "SYNTHETIC" in spec.title


def test_attrition_overview_bar():
    ev = {"id": "E2", "tool": "attrition_insight", "cited": True,
          "rows": [{"segment_type": "department", "segment": "Elections", "avg_predicted_risk": 0.40},
                   {"segment_type": "age_band", "segment": "<25", "avg_predicted_risk": 0.25}]}
    [spec] = charts_for_attrition(ev)
    assert "Elections (department)" in spec.figure.data[0].y and "SYNTHETIC" in spec.title


def test_table_inference():
    years = pd.DataFrame({"fiscal_year": [2024, 2025, 2026], "n": [10, 12, 15]})
    assert chart_for_table(years).figure.data[0].type == "scatter"
    cats = pd.DataFrame({"department": ["A", "B"], "n": [3, 5]})
    assert chart_for_table(cats).figure.data[0].type == "bar"
    assert chart_for_table(pd.DataFrame({"n": [42930]})).kind == "kpi"
    assert chart_for_table(pd.DataFrame({"a": ["x", "x"], "b": ["y", "z"]})) is None


def test_uncited_and_docs_evidence_have_no_chart():
    ev = [{"id": "E1", "tool": "search_docs", "cited": True},
          {"id": "E2", "tool": "attrition_insight", "cited": False,
           "rows": [{"segment_type": "level", "segment": "L1", "avg_predicted_risk": 0.2}]}]
    assert charts_for_evidence(ev, {"metrics": {}, "models": {}}) == []


def test_single_attrition_segment_is_kpi_not_one_bar():
    ev = {"id": "E1", "tool": "attrition_insight", "cited": True,
          "rows": [{"segment_type": "department", "segment": "Elections", "avg_predicted_risk": 0.398,
                    "high_risk_count": 204, "expected_leavers": 133.7}]}
    [spec] = charts_for_attrition(ev)
    assert spec.kind == "kpi" and ("Elections · predicted risk (SYNTHETIC)", "39.8%") in spec.kpis
    assert ("Elections · expected leavers", "134") in spec.kpis
