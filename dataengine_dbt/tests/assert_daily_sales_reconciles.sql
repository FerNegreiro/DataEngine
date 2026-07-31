with fact_daily as (
    select
        date(order_date) as sales_date,
        sum(quantity) as total_items_sold,
        count(distinct product_id) as distinct_products,
        round(sum(line_gross_amount), 2) as gross_revenue,
        round(sum(line_discount_amount), 2) as item_discount_amount
    from {{ ref('fct_sales') }}
    where is_realized_sale
    group by sales_date
),

order_daily as (
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
),

expected as (
    select
        order_daily.sales_date,
        order_daily.total_orders,
        order_daily.unique_customers,
        fact_daily.total_items_sold,
        fact_daily.distinct_products,
        fact_daily.gross_revenue,
        fact_daily.item_discount_amount,
        order_daily.shipping_amount,
        order_daily.order_discount_amount,
        order_daily.total_revenue,
        order_daily.average_order_value
    from order_daily
    inner join fact_daily
        on order_daily.sales_date = fact_daily.sales_date
)

select
    coalesce(expected.sales_date, actual.sales_date) as sales_date
from expected
full outer join {{ ref('mart_daily_sales') }} as actual
    on expected.sales_date = actual.sales_date
where
    expected.sales_date is null
    or actual.sales_date is null
    or expected.total_orders != actual.total_orders
    or expected.unique_customers != actual.unique_customers
    or expected.total_items_sold != actual.total_items_sold
    or expected.distinct_products != actual.distinct_products
    or abs(expected.gross_revenue - actual.gross_revenue) > 0.01
    or abs(expected.item_discount_amount - actual.item_discount_amount) > 0.01
    or abs(expected.shipping_amount - actual.shipping_amount) > 0.01
    or abs(expected.order_discount_amount - actual.order_discount_amount) > 0.01
    or abs(expected.total_revenue - actual.total_revenue) > 0.01
    or abs(expected.average_order_value - actual.average_order_value) > 0.01
