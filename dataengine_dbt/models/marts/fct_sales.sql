with order_items as (
    select
        order_item_id,
        order_id,
        product_id,
        quantity,
        unit_price,
        unit_cost,
        discount_percentage,
        line_gross_amount,
        line_discount_amount,
        line_total,
        line_cost_amount,
        line_margin_before_order_discount
    from {{ ref('int_order_items_enriched') }}
),

orders as (
    select
        order_id,
        customer_id,
        order_date,
        order_status,
        shipping_cost,
        discount_amount,
        order_total
    from {{ ref('stg_orders') }}
)

select
    order_items.order_item_id,
    order_items.order_id,
    orders.customer_id,
    order_items.product_id,
    orders.order_date,
    orders.order_status,
    orders.order_status != 'Cancelado' as is_realized_sale,
    order_items.quantity,
    order_items.unit_price,
    order_items.unit_cost,
    order_items.discount_percentage,
    order_items.line_gross_amount,
    order_items.line_discount_amount,
    order_items.line_total as item_total,
    order_items.line_cost_amount,
    order_items.line_margin_before_order_discount,
    orders.shipping_cost as order_shipping_cost,
    orders.discount_amount as order_discount_amount,
    orders.order_total
from order_items
inner join orders
    on order_items.order_id = orders.order_id
