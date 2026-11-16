"""Attrition feature set (single definition, reused by train and score).

Source: mart.fct_employee_snapshot (pay REAL, HRIS SYNTHETIC). Tenure is measured at the START of the
fiscal year so the termination date cannot leak into the features. Gender is excluded on purpose.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

NUMERIC = ["tenure_at_start", "compa_ratio", "overtime_ratio", "performance_rating", "level_num",
           "age_band_num", "log_salary", "is_temporary"]
CATEGORICAL = ["organization_group"]
LABEL = "is_terminated"
AGE_ORDER = {"<25": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
FEATURE_LABELS = {
    "tenure_at_start": "Tenure (years at start of FY)", "compa_ratio": "Pay vs job median (compa-ratio)",
    "overtime_ratio": "Overtime / salary", "performance_rating": "Performance rating",
    "level_num": "Level", "age_band_num": "Age band", "log_salary": "Salary level",
    "is_temporary": "Temporary employment", "organization_group": "Organization group",
}

QUERY = """
select s.employee_key, s.fiscal_year, d.department, d.organization_group, j.job_family, s.level, s.age_band,
       s.gender, s.employment_type, s.total_salary,
       date_diff('day', s.hire_date, make_date(s.fiscal_year - 1, 7, 1)) / 365.25 as tenure_at_start,
       s.compa_ratio_internal as compa_ratio, coalesce(s.overtime_ratio, 0) as overtime_ratio,
       s.performance_rating, s.is_terminated
from mart.fct_employee_snapshot s
left join mart.dim_department d using (department_code)
left join mart.dim_job j using (job_code)
order by s.employee_key  -- fixed row order keeps the seeded train/test split reproducible
"""


def load_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(QUERY).df()
    df["level_num"] = df["level"].str[1:].astype(int)
    df["age_band_num"] = df["age_band"].map(AGE_ORDER).astype(int)
    df["log_salary"] = np.log1p(df["total_salary"].clip(lower=0))
    df["is_temporary"] = df["employment_type"].fillna("").str.startswith("Temporary").astype(int)
    df[LABEL] = df[LABEL].astype(int)
    return df


def design_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Numeric features + one-hot organization group; align to `columns` when scoring."""
    X = pd.concat([df[NUMERIC].astype(float),
                   pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, dtype=float)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def base_feature(column: str) -> str:
    """Map a one-hot column back to its source feature (for grouped SHAP importance)."""
    return next((c for c in CATEGORICAL if column.startswith(c + "_")), column)
