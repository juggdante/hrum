
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    compensation_record_id as unique_field,
    count(*) as n_records

from "hrum"."mart"."fct_compensation"
where compensation_record_id is not null
group by compensation_record_id
having count(*) > 1



  
  
      
    ) dbt_internal_test