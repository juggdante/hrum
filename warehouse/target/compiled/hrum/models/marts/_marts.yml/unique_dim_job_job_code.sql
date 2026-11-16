
    
    

select
    job_code as unique_field,
    count(*) as n_records

from "hrum"."mart"."dim_job"
where job_code is not null
group by job_code
having count(*) > 1


