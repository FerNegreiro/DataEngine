select
    order_item_id,
    order_id,
    product_id,
    quantity,
    unit_price,
    unit_cost,
    discount_percentage,
    line_total
from {{ source('dataengine', 'order_items') }}
