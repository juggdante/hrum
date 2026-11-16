-- One row per employee in the snapshot fiscal year.
-- MIXED: department/job/comp are REAL; every HRIS field (hire/termination, level, performance,
-- gender, age band, location, tenure, attrition flag) is SYNTHETIC -> is_synthetic = true.
with comp as (
    select *
    from "hrum"."mart"."fct_compensation"
    where is_paid
    qualify row_number() over (partition by employee_key, fiscal_year order by total_compensation desc) = 1
)

select
    h.employee_key,
    h.snapshot_fiscal_year                                       as fiscal_year,
    c.department_code,
    c.job_code,
    c.employment_type,
    c.total_salary,
    c.overtime,
    c.total_benefits,
    c.total_compensation,
    c.hours,
    c.total_salary / median(c.total_salary) over (partition by c.job_code)  as compa_ratio_internal,
    case when c.total_salary > 0 then c.overtime / c.total_salary end       as overtime_ratio,
    h.hire_date,
    h.termination_date,
    h.termination_reason,
    round(date_diff('day', h.hire_date,
          coalesce(h.termination_date, make_date(h.snapshot_fiscal_year, 6, 30))) / 365.25, 2) as tenure_years,
    h.level,
    h.performance_rating,
    h.gender,
    h.age_band,
    h.location,
    h.termination_date is not null                               as is_terminated,
    h.termination_reason like 'Voluntary%'                       as is_voluntary_termination,
    true                                                         as is_synthetic
from "hrum"."stg"."stg_synthetic__hris" h
join comp c
  on c.employee_key = h.employee_key
 and c.fiscal_year = h.snapshot_fiscal_year