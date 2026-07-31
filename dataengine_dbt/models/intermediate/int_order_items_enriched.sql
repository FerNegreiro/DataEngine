with order_items as (
    select
        order_item_id,
        order_id,
        product_id,
        quantity,
        unit_price,
        unit_cost,
        discount_percentage,
        line_total
    from {{ ref('stg_order_items') }}
),

products as (
    select
        product_id,
        product_name,
        category
    from {{ ref('stg_products') }}
)

select
    order_items.order_item_id,
    order_items.order_id,
    order_items.product_id,
    products.product_name,
    products.category,
    order_items.quantity,
    order_items.unit_price,
    order_items.unit_cost,
    order_items.discount_percentage,
    round(order_items.quantity * order_items.unit_price, 2) as line_gross_amount,
    round(
        (order_items.quantity * order_items.unit_price) - order_items.line_total,
        2
    ) as line_discount_amount,
    order_items.line_total,
    round(order_items.quantity * order_items.unit_cost, 2) as line_cost_amount,
    round(
        order_items.line_total - (order_items.quantity * order_items.unit_cost),
        2
    ) as line_margin_before_order_discount
from order_items
inner join products
    on order_items.product_id = products.product_id
