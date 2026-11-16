
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        is_synthetic as value_field,
        count(*) as n_records

    from "hrum"."stg"."stg_synthetic__hris"
    group by is_synthetic

)

select *
from all_values
where value_field not in (
    'True'
)



  
  
      
    ) dbt_internal_test