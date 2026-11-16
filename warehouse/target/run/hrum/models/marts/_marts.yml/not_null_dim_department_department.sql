
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select department
from "hrum"."mart"."dim_department"
where department is null



  
  
      
    ) dbt_internal_test