
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select year
from "hrum"."stg"."stg_sf_comp__compensation"
where year is null



  
  
      
    ) dbt_internal_test