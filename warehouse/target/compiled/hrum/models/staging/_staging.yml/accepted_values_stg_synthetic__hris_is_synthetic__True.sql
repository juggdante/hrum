
    
    

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


