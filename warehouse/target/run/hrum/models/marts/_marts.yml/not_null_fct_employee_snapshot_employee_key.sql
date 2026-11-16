
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select employee_key
from "hrum"."mart"."fct_employee_snapshot"
where employee_key is null



  
  
      
    ) dbt_internal_test