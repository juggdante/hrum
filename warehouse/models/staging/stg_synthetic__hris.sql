-- SYNTHETIC HRIS snapshot (ingestion/synthetic/generate.py). Typing only.
select
    employee_key,
    cast(snapshot_fiscal_year as integer)  as snapshot_fiscal_year,
    cast(hire_date as date)                as hire_date,
    cast(termination_date as date)         as termination_date,
    termination_reason,
    level,
    cast(performance_rating as integer)    as performance_rating,
    gender,
    age_band,
    location,
    is_synthetic
from {{ source('raw', 'synthetic_hris') }}
