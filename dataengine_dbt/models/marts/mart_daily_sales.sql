with item_metrics as (
    select
        date(order_date) as sales_date,
        sum(quantity) as total_items_sold,
        count(distinct product_id) as distinct_products,
        round(sum(line_gross_amount), 2) as gross_revenue,
        round(sum(line_discount_amount), 2) as item_discount_amount,
        round(sum(item_total), 2) as item_revenue_before_order_discount
    from {{ ref('fct_sales') }}
    where is_realized_sale
    group by sales_date
),

order_metrics as (
    select
        date(order_date) as sales_date,
        count(*) as total_orders,
        count(distinct customer_id) as unique_customers,
        round(sum(shipping_cost), 2) as shipping_amount,
        round(sum(order_discount_amount), 2) as order_discount_amount,
        round(sum(order_total), 2) as total_revenue,
        round(avg(order_total), 2) as average_order_value
    from {{ ref('int_orders_enriched') }}
    where order_status != 'Cancelado'
    group by sales_date
)

select
    order_metrics.sales_date,
    order_metrics.total_orders,
    order_metrics.unique_customers,
    item_metrics.total_items_sold,
    item_metrics.distinct_products,
    item_metrics.gross_revenue,
    item_metrics.item_discount_amount,
    order_metrics.order_discount_amount,
    round(
        item_metrics.item_discount_amount + order_metrics.order_discount_amount,
        2
    ) as total_discount_amount,
    item_metrics.item_revenue_before_order_discount,
    order_metrics.shipping_amount,
    order_metrics.total_revenue,
    order_metrics.average_order_value
from order_metrics
inner join item_metrics
    on order_metrics.sales_date = item_metrics.sales_date
