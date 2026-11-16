
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with child as (
    select department_code as from_field
    from "hrum"."mart"."fct_compensation"
    where department_code is not null
),

parent as (
    select department_code as to_field
    from "hrum"."mart"."dim_department"
)

select
    from_field

from child
left join parent
    on child.from_field = parent.to_field

where parent.to_field is null



  
  
      
    ) dbt_internal_test