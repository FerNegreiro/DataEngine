with customers as (
    select
        customer_id,
        full_name,
        city,
        state,
        region,
        customer_segment,
        is_active
    from {{ ref('dim_customers') }}
),

sales_reference as (
    select max(date(order_date)) as reference_date
    from {{ ref('int_orders_enriched') }}
    where order_status != 'Cancelado'
),

customer_orders as (
    select
        customer_id,
        count(*) as total_orders,
        sum(item_quantity) as total_items_purchased,
        round(sum(order_total), 2) as total_spend,
        round(avg(order_total), 2) as average_order_value,
        min(date(order_date)) as first_purchase_date,
        max(date(order_date)) as last_purchase_date
    from {{ ref('int_orders_enriched') }}
    where order_status != 'Cancelado'
    group by customer_id
)

select
    customers.customer_id,
    customers.full_name,
    customers.city,
    customers.state,
    customers.region,
    customers.customer_segment,
    customers.is_active,
    coalesce(customer_orders.total_orders, 0) as total_orders,
    coalesce(
        customer_orders.total_items_purchased,
        0
    ) as total_items_purchased,
    coalesce(customer_orders.total_spend, 0) as total_spend,
    coalesce(customer_orders.average_order_value, 0) as average_order_value,
    customer_orders.first_purchase_date,
    customer_orders.last_purchase_date,
    date_diff(
        sales_reference.reference_date,
        customer_orders.last_purchase_date,
        day
    ) as days_since_last_purchase,
    sales_reference.reference_date as sales_reference_date
from customers
cross join sales_reference
left join customer_orders
    on customers.customer_id = customer_orders.customer_id
