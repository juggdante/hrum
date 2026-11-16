"""Attrition model insight tool: aggregated predicted risk + drivers (SYNTHETIC labels)."""
from __future__ import annotations

from typing import Any

from agent.core.context import RunContext
from agent.tools import Tool

SEGMENT_TYPES = ["organization_group", "department", "job_family", "level", "age_band", "gender"]
CAVEATS = [
    "Uses SYNTHETIC HRIS fields: demonstrates capability, not a real finding about SF.",
    ("Attrition risk comes from an XGBoost model trained on synthetic labels; drivers mirror the data "
    "generator rules. Gender is not a model input."),
    "Segments with fewer than 5 employees are not published.",
]


def _attrition_insight(args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    con = ctx.connection()
    seg_type = args.get("segment_type", "all")
    top_n = int(args.get("top_n", 10))
    cols = """segment_type, segment, employee_count, avg_predicted_risk, high_risk_count, expected_leavers,
              observed_turnover_rate, top_driver_1, top_driver_2, top_driver_3"""
    if seg_type == "all":  # overview: the highest-risk segment of every segment type, in one call
        rows = con.execute(
            f"""select {cols} from mart.ml_attrition_risk_segment
                qualify row_number() over (partition by segment_type order by avg_predicted_risk desc) = 1
                order by avg_predicted_risk desc""").df()
    else:
        rows = con.execute(
            f"""select {cols} from mart.ml_attrition_risk_segment where segment_type = $t
                order by avg_predicted_risk desc limit $n""", {"t": seg_type, "n": top_n}).df()
    drivers = con.execute(
        "select label as driver, mean_abs_shap, test_roc_auc, test_top_decile_lift "
        "from mart.ml_attrition_drivers order by mean_abs_shap desc").df()
    return {
        "segment_type": seg_type,
        "rows": rows.to_dict("records"),
        "global_drivers": drivers[["driver", "mean_abs_shap"]].to_dict("records"),
        "model_quality": {"test_roc_auc": float(drivers["test_roc_auc"].iloc[0]),
                          "test_top_decile_lift": float(drivers["test_top_decile_lift"].iloc[0])},
        "definitions": {
            "avg_predicted_risk": "Mean predicted probability of leaving within the fiscal year.",
            "high_risk_count": "Employees with predicted risk >= 0.30.",
            "expected_leavers": "Sum of predicted probabilities (expected number of leavers).",
            "observed_turnover_rate": "Actual (synthetic) share who left in FY2026.",
        },
        "caveats": CAVEATS,
        "row_count": len(rows),
    }


ATTRITION_INSIGHT = Tool(
    name="attrition_insight",
    description="Attrition risk from the ML model (SYNTHETIC data): average predicted risk, high-risk counts, "
                "expected leavers and top risk drivers per segment, plus global drivers and model quality. "
                "Call ONCE. Default segment_type 'all' returns the highest-risk segment of every segment type; "
                "pass one segment_type only to rank segments within that type.",
    parameters={"type": "object",
                "properties": {"segment_type": {"type": "string", "enum": ["all", *SEGMENT_TYPES]},
                               "top_n": {"type": "integer", "minimum": 1, "maximum": 50}},
                "additionalProperties": False},
    func=_attrition_insight, roles=frozenset({"chro", "analyst"}),
)
