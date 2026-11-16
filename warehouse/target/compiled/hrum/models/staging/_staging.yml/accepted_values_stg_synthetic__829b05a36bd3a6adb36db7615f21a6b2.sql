
    
    

with all_values as (

    select
        performance_rating as value_field,
        count(*) as n_records

    from "hrum"."stg"."stg_synthetic__hris"
    group by performance_rating

)

select *
from all_values
where value_field not in (
    1,2,3,4,5
)


