select
    'stg_orders' as model_name,
    'shipping_cost' as column_name,
    order_id as record_id,
    shipping_cost as invalid_value
from {{ ref('stg_orders') }}
where shipping_cost < 0

union all

select
    'stg_orders',
    'discount_amount',
    order_id,
    discount_amount
from {{ ref('stg_orders') }}
where discount_amount < 0

union all

select
    'stg_orders',
    'order_total',
    order_id,
    order_total
from {{ ref('stg_orders') }}
where order_total < 0

union all

select
    'stg_order_items',
    'unit_price',
    order_item_id,
    unit_price
from {{ ref('stg_order_items') }}
where unit_price < 0

union all

select
    'stg_order_items',
    'unit_cost',
    order_item_id,
    unit_cost
from {{ ref('stg_order_items') }}
where unit_cost < 0

union all

select
    'stg_order_items',
    'line_total',
    order_item_id,
    line_total
from {{ ref('stg_order_items') }}
where line_total < 0

union all

select
    'stg_products',
    'unit_price',
    product_id,
    unit_price
from {{ ref('stg_products') }}
where unit_price < 0

union all

select
    'stg_products',
    'unit_cost',
    product_id,
    unit_cost
from {{ ref('stg_products') }}
where unit_cost < 0
