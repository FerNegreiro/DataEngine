select
    order_item_id,
    count(*) as row_count
from {{ ref('fct_sales') }}
group by order_item_id
having count(*) != 1
