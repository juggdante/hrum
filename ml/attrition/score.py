"""Score every employee and publish AGGREGATED risk to the mart (no person-level output leaves this script).

Writes (all is_synthetic = true):
  mart.ml_attrition_risk_segment   one row per (segment_type, segment) with >= min_group_size employees
  mart.ml_attrition_drivers        global mean |SHAP| per feature + model metrics
Run after train:  python -m ml.attrition.score
"""
from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from common.config import db_path, load_config
from ml.attrition.train import REGISTRY
from ml.features.attrition import FEATURE_LABELS, base_feature, design_matrix, load_frame

log = logging.getLogger(__name__)
SEGMENT_TYPES = ["organization_group", "department", "job_family", "level", "age_band", "gender"]
HIGH_RISK = 0.30


def top_drivers(shap_df: pd.DataFrame, n: int = 3) -> list[str]:
    """Features that push risk UP the most on average in this segment."""
    mean = shap_df.mean().sort_values(ascending=False)
    return [FEATURE_LABELS.get(f, f) for f, v in mean.items() if v > 0][:n]


def segment_table(df: pd.DataFrame, shap_grouped: pd.DataFrame, min_group: int) -> pd.DataFrame:
    rows = []
    for seg_type in SEGMENT_TYPES:
        for seg, idx in df.groupby(seg_type).groups.items():
            if len(idx) < min_group:
                continue
            part = df.loc[idx]
            drivers = top_drivers(shap_grouped.loc[idx])
            rows.append({
                "segment_type": seg_type, "segment": str(seg), "employee_count": len(idx),
                "avg_predicted_risk": round(float(part["risk"].mean()), 4),
                "high_risk_count": int((part["risk"] >= HIGH_RISK).sum()),
                "expected_leavers": round(float(part["risk"].sum()), 1),
                "observed_turnover_rate": round(float(part["is_terminated"].mean()), 4),
                "top_driver_1": drivers[0] if len(drivers) > 0 else None,
                "top_driver_2": drivers[1] if len(drivers) > 1 else None,
                "top_driver_3": drivers[2] if len(drivers) > 2 else None,
                "is_synthetic": True,
            })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    card = json.loads((REGISTRY / "model_card.json").read_text())
    model = xgb.XGBClassifier()
    model.load_model(REGISTRY / "model.json")
    min_group = int(load_config("guardrails")["privacy"]["min_group_size"])

    with duckdb.connect(str(db_path())) as con:
        df = load_frame(con)
        X = design_matrix(df, card["features"])
        df["risk"] = model.predict_proba(X)[:, 1]
        values = shap.TreeExplainer(model).shap_values(X)
        shap_grouped = pd.DataFrame(values, columns=X.columns, index=df.index)
        shap_grouped = shap_grouped.T.groupby([base_feature(c) for c in X.columns]).sum().T
        segments = segment_table(df, shap_grouped, min_group)
        drivers = pd.DataFrame(card["global_importance"]).assign(  # noqa: F841 - read by DuckDB SQL below
            model=card["name"], trained_at=card["trained_at"],
            test_roc_auc=card["metrics_test"]["xgboost"]["roc_auc"],
            test_top_decile_lift=card["metrics_test"]["xgboost"]["top_decile_lift"], is_synthetic=True)
        con.execute("create or replace table mart.ml_attrition_risk_segment as select * from segments")
        con.execute("create or replace table mart.ml_attrition_drivers as select * from drivers")
    log.info("scored %s employees (mean risk %.3f); %s segments published",
             f"{len(df):,}", float(np.mean(df["risk"])), len(segments))


if __name__ == "__main__":
    main()
