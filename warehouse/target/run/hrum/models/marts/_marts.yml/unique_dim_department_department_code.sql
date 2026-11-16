
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    department_code as unique_field,
    count(*) as n_records

from "hrum"."mart"."dim_department"
where department_code is not null
group by department_code
having count(*) > 1



  
  
      
    ) dbt_internal_test