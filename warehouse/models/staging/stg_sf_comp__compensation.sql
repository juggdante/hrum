-- Cleaning only: typed columns, snake_case names, hashed person key. No business filters here.
with source as (
    select * from {{ source('raw', 'sf_employee_compensation') }}
)

select
    -- grain: one row per (year_type, year, employee, job, department) pay record
    md5(concat_ws('|', "Year Type", "Year", "Employee Name", "Job Code", "Department Code",
                  "Total Compensation", "Hours"))              as compensation_record_id,
    -- PII: the name is hashed here and never selected downstream as text
    md5('hrum|' || "Employee Name")                           as employee_key,
    "Year Type"                                               as year_type,
    cast("Year" as integer)                                   as year,
    "Organization Group Code"                                 as organization_group_code,
    "Organization Group"                                      as organization_group,
    -- 2 FY2017 rows have no department in the source; keep them so city totals reconcile
    coalesce("Department Code", 'UNK')                        as department_code,
    coalesce("Department", 'Unknown (not reported)')          as department,
    "Union Code"                                              as union_code,
    "Union"                                                   as union_name,
    "Job Family Code"                                         as job_family_code,
    "Job Family"                                              as job_family,
    "Job Code"                                                as job_code,
    "Job"                                                     as job,
    "Employment Type"                                         as employment_type,
    {{ parse_amount('Salaries') }}                            as salaries,
    {{ parse_amount('Overtime') }}                            as overtime,
    {{ parse_amount('Other Salaries') }}                      as other_salaries,
    {{ parse_amount('Total Salary') }}                        as total_salary,
    {{ parse_amount('Retirement') }}                          as retirement,
    {{ parse_amount('Health and Dental') }}                   as health_and_dental,
    {{ parse_amount('Other Benefits') }}                      as other_benefits,
    {{ parse_amount('Total Benefits') }}                      as total_benefits,
    {{ parse_amount('Total Compensation') }}                  as total_compensation,
    {{ parse_amount('Hours') }}                               as hours,
    _loaded_at,
    _source_file,
    false                                                     as is_synthetic
from source
