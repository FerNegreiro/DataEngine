with orders as (
    select
        order_id,
        customer_id,
        order_date,
        order_status,
        payment_method,
        sales_channel,
        shipping_cost,
        discount_amount,
        order_total,
        delivery_date
    from {{ ref('stg_orders') }}
),

customers as (
    select
        customer_id,
        full_name,
        city,
        state,
        region
    from {{ ref('stg_customers') }}
),

item_metrics as (
    select
        order_id,
        count(*) as order_item_count,
        sum(quantity) as item_quantity,
        count(distinct product_id) as distinct_products,
        round(sum(line_gross_amount), 2) as gross_amount,
        round(sum(line_discount_amount), 2) as item_discount_amount,
        round(sum(line_total), 2) as items_net_amount,
        round(sum(line_cost_amount), 2) as item_cost_amount
    from {{ ref('int_order_items_enriched') }}
    group by order_id
)

select
    orders.order_id,
    orders.customer_id,
    customers.full_name as customer_name,
    customers.city as customer_city,
    customers.state as customer_state,
    customers.region as customer_region,
    orders.order_date,
    orders.delivery_date,
    orders.order_status,
    orders.payment_method,
    orders.sales_channel,
    item_metrics.order_item_count,
    item_metrics.item_quantity,
    item_metrics.distinct_products,
    item_metrics.gross_amount,
    item_metrics.item_discount_amount,
    item_metrics.items_net_amount,
    item_metrics.item_cost_amount,
    orders.shipping_cost,
    orders.discount_amount as order_discount_amount,
    round(
        item_metrics.item_discount_amount + orders.discount_amount,
        2
    ) as total_discount_amount,
    orders.order_total
from orders
inner join customers
    on orders.customer_id = customers.customer_id
inner join item_metrics
    on orders.order_id = item_metrics.order_id
