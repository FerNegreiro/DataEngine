select
    order_item_id,
    quantity,
    discount_percentage
from {{ ref('stg_order_items') }}
where
    quantity <= 0
    or discount_percentage < 0
    or discount_percentage > 100
