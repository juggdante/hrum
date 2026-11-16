
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select employee_key
from "hrum"."stg"."stg_synthetic__hris"
where employee_key is null



  
  
      
    ) dbt_internal_test