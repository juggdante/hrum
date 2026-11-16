
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select compensation_record_id
from "hrum"."mart"."fct_compensation"
where compensation_record_id is null



  
  
      
    ) dbt_internal_test