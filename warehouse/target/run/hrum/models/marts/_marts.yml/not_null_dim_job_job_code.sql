
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select job_code
from "hrum"."mart"."dim_job"
where job_code is null



  
  
      
    ) dbt_internal_test