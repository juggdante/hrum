
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select department_code
from "hrum"."mart"."dim_department"
where department_code is null



  
  
      
    ) dbt_internal_test