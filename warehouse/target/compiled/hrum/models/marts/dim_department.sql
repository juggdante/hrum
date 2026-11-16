-- Latest name per department code (names occasionally change across years).
select
    department_code,
    arg_max(department, year)          as department,
    arg_max(organization_group, year)  as organization_group,
    min(year)                          as first_fiscal_year,
    max(year)                          as last_fiscal_year
from "hrum"."stg"."stg_sf_comp__compensation"
where year_type = 'Fiscal'
group by department_code