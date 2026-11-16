
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_compensation
from "hrum"."mart"."fct_compensation"
where total_compensation is null



  
  
      
    ) dbt_internal_test