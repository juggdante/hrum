
    

    create  table
      "hrum"."mart"."dim_job__dbt_tmp"
  
    
    as (
      -- Latest title per job code, with its job family.
select
    job_code,
    arg_max(job, year)                 as job,
    arg_max(job_family_code, year)     as job_family_code,
    arg_max(job_family, year)          as job_family,
    min(year)                          as first_fiscal_year,
    max(year)                          as last_fiscal_year
from "hrum"."stg"."stg_sf_comp__compensation"
where year_type = 'Fiscal'
group by job_code
    );
    
  