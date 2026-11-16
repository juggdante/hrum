
    
    

select
    employee_key as unique_field,
    count(*) as n_records

from "hrum"."stg"."stg_synthetic__hris"
where employee_key is not null
group by employee_key
having count(*) > 1


