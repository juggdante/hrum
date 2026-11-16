-- One row per FISCAL-year pay record (Calendar rows are a second view of the same pay; see data_dictionary.md).
-- No names: employee_key is a hash. REAL data (is_synthetic = false).
select
    compensation_record_id,
    employee_key,
    year                                                    as fiscal_year,
    department_code,
    job_code,
    employment_type,
    salaries,
    overtime,
    other_salaries,
    total_salary,
    retirement,
    health_and_dental,
    other_benefits,
    total_benefits,
    total_compensation,
    hours,
    hours >= 1820                   as is_full_time,
    total_compensation < 0                                  as is_adjustment,
    total_salary > 0                                        as is_paid,
    is_synthetic
from "hrum"."stg"."stg_sf_comp__compensation"
where year_type = 'Fiscal'