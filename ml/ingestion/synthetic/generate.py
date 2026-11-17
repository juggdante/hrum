"""Generate the SYNTHETIC HRIS layer -> DuckDB raw.synthetic_hris.

Population = REAL FY<snapshot> paid SF Comp records (department, job, comp come from the real data).
Every HRIS field added here is SYNTHETIC: hire_date, termination_date, level, performance_rating,
gender, age_band, location. Attrition is simulated from a documented logistic rule so the ML layer
has learnable signal; model output on it demonstrates capability, NOT a finding about San Francisco.

Requires `make dbt-run` (stg_sf_comp__compensation) to exist first. Deterministic (fixed seed).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from common.config import db_path

log = logging.getLogger(__name__)
SEED = 42
SNAPSHOT_FY = 2026
FY_START, FY_END = date(SNAPSHOT_FY - 1, 7, 1), date(SNAPSHOT_FY, 6, 30)
LOCATIONS = ["Civic Center", "Mission Bay", "SoMa", "Bayview", "Sunset", "Richmond", "Airport (SFO)"]
AGE_BANDS = ["<25", "25-34", "35-44", "45-54", "55+"]
# Share female by job family (rough public-sector pattern); unlisted families default to 0.45.
FEMALE_SHARE = {"Nursing": 0.85, "Police Services": 0.20, "Fire Services": 0.15, "Clerical, Secretarial & Steno": 0.75,
                "Street Transit": 0.30, "Journeyman Trade": 0.08, "Human Services": 0.70, "Library": 0.65}
# Logistic attrition rule (log-odds). Documented in docs/data_dictionary.md.
ATTRITION_COEF = {"intercept": -2.6, "compa_below_0_9": 0.8, "perf_low": 0.9, "perf_high": -0.5,
                  "tenure_lt_2": 0.9, "tenure_gt_15": -0.6, "overtime_ratio_gt_0_25": 0.5, "temporary": 0.7}


def load_population(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per employee_key in the snapshot FY (highest-paid record if a key repeats)."""
    return con.execute(
        """
        select employee_key, department_code, job_family, job_code, employment_type,
               total_salary, overtime, total_compensation, hours
        from stg.stg_sf_comp__compensation
        where year_type = 'Fiscal' and year = $fy and total_salary > 0
        qualify row_number() over (partition by employee_key
                                   order by total_compensation desc, compensation_record_id) = 1
        order by employee_key  -- fixed row order: the seeded RNG must hit the same rows on every run
        """,
        {"fy": SNAPSHOT_FY},
    ).df()


def add_hris_fields(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    temporary = df["employment_type"].fillna("").str.startswith("Temporary").to_numpy()
    # tenure: permanent staff stay longer
    tenure_years = np.where(temporary, rng.gamma(1.5, 1.2, n), rng.gamma(2.0, 4.5, n)).clip(0.1, 40)
    df["hire_date"] = [FY_START - timedelta(days=int(t * 365.25)) for t in tenure_years]
    df["tenure_years"] = tenure_years.round(2)

    age = (22 + tenure_years + rng.normal(10, 6, n)).clip(18, 70)
    df["age_band"] = pd.cut(age, [0, 25, 35, 45, 55, 200], right=False, labels=AGE_BANDS).astype(str)

    p_female = df["job_family"].map(FEMALE_SHARE).fillna(0.45).to_numpy()
    df["gender"] = np.where(rng.random(n) < p_female, "F", "M")

    # level = comp quintile within job family (L1 lowest .. L5 highest)
    pct = df.groupby("job_family")["total_salary"].rank(pct=True).fillna(0.5)
    df["level"] = "L" + np.ceil(pct * 5).clip(1, 5).astype(int).astype(str)

    df["performance_rating"] = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.50, 0.22, 0.08])
    df["location"] = rng.choice(LOCATIONS, n)

    job_median = df.groupby("job_code")["total_salary"].transform("median")
    df["compa_ratio"] = (df["total_salary"] / job_median).round(3)
    overtime_ratio = (df["overtime"] / df["total_salary"]).fillna(0)

    c = ATTRITION_COEF
    logit = (c["intercept"]
             + c["compa_below_0_9"] * (df["compa_ratio"] < 0.9)
             + c["perf_low"] * (df["performance_rating"] <= 2)
             + c["perf_high"] * (df["performance_rating"] >= 5)
             + c["tenure_lt_2"] * (tenure_years < 2)
             + c["tenure_gt_15"] * (tenure_years > 15)
             + c["overtime_ratio_gt_0_25"] * (overtime_ratio > 0.25)
             + c["temporary"] * temporary)
    left = rng.random(n) < 1 / (1 + np.exp(-logit.to_numpy()))
    days = (FY_END - FY_START).days
    df["termination_date"] = [FY_START + timedelta(days=int(rng.integers(0, days))) if x else None for x in left]
    df["termination_reason"] = np.where(
        left, rng.choice(["Voluntary - other employer", "Voluntary - personal", "Retirement", "Involuntary"],
                         n, p=[0.45, 0.25, 0.18, 0.12]), None)
    df["snapshot_fiscal_year"] = SNAPSHOT_FY
    df["is_synthetic"] = True
    return df


COLUMNS = ["employee_key", "snapshot_fiscal_year", "hire_date", "termination_date", "termination_reason",
           "level", "performance_rating", "gender", "age_band", "location", "is_synthetic"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(SEED)
    with duckdb.connect(str(db_path())) as con:
        df = add_hris_fields(load_population(con), rng)[COLUMNS]
        con.execute("create schema if not exists raw")
        con.execute("create or replace table raw.synthetic_hris as select * from df")
    log.info("synthetic_hris: %s employees, %.1f%% terminated in FY%s",
             f"{len(df):,}", 100 * df["termination_date"].notna().mean(), SNAPSHOT_FY)


if __name__ == "__main__":
    main()
