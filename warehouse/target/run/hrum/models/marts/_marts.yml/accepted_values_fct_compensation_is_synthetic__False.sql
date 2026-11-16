
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        is_synthetic as value_field,
        count(*) as n_records

    from "hrum"."mart"."fct_compensation"
    group by is_synthetic

)

select *
from all_values
where value_field not in (
    'False'
)



  
  
      
    ) dbt_internal_test