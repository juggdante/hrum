
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        year_type as value_field,
        count(*) as n_records

    from "hrum"."stg"."stg_sf_comp__compensation"
    group by year_type

)

select *
from all_values
where value_field not in (
    'Fiscal','Calendar'
)



  
  
      
    ) dbt_internal_test