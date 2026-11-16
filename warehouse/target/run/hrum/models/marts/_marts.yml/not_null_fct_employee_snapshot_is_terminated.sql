
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select is_terminated
from "hrum"."mart"."fct_employee_snapshot"
where is_terminated is null



  
  
      
    ) dbt_internal_test