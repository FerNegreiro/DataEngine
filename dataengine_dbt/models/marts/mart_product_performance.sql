with products as (
    select
        product_id,
        product_name,
        category,
        stock_quantity,
        minimum_stock,
        is_active
    from {{ ref('dim_products') }}
),

sales_reference as (
    select max(date(order_date)) as reference_date
    from {{ ref('fct_sales') }}
    where is_realized_sale
),

lifetime_sales as (
    select
        product_id,
        count(distinct order_id) as total_orders,
        sum(quantity) as total_items_sold,
        count(distinct customer_id) as unique_customers,
        round(sum(item_total), 2) as item_revenue_before_order_discount,
        round(sum(line_cost_amount), 2) as item_cost_amount,
        round(sum(line_margin_before_order_discount), 2)
            as margin_before_order_discount
    from {{ ref('fct_sales') }}
    where is_realized_sale
    group by product_id
),

recent_sales as (
    select
        sales.product_id,
        sum(sales.quantity) as items_sold_last_30_days
    from {{ ref('fct_sales') }} as sales
    cross join sales_reference
    where
        sales.is_realized_sale
        and date(sales.order_date) between
            date_sub(sales_reference.reference_date, interval 29 day)
            and sales_reference.reference_date
    group by sales.product_id
)

select
    products.product_id,
    products.product_name,
    products.category,
    coalesce(lifetime_sales.total_orders, 0) as total_orders,
    coalesce(lifetime_sales.total_items_sold, 0) as total_items_sold,
    coalesce(lifetime_sales.unique_customers, 0) as unique_customers,
    coalesce(
        lifetime_sales.item_revenue_before_order_discount,
        0
    ) as item_revenue_before_order_discount,
    coalesce(lifetime_sales.item_cost_amount, 0) as item_cost_amount,
    coalesce(
        lifetime_sales.margin_before_order_discount,
        0
    ) as margin_before_order_discount,
    products.stock_quantity as current_stock_quantity,
    products.minimum_stock,
    coalesce(recent_sales.items_sold_last_30_days, 0)
        as items_sold_last_30_days,
    sales_reference.reference_date as sales_reference_date,
    products.is_active,
    case
        when products.stock_quantity <= products.minimum_stock
            then 'below_minimum'
        when
            coalesce(recent_sales.items_sold_last_30_days, 0)
            >= products.stock_quantity
            and coalesce(recent_sales.items_sold_last_30_days, 0) > 0
            then 'at_risk'
        else 'adequate'
    end as stock_risk_status
from products
cross join sales_reference
left join lifetime_sales
    on products.product_id = lifetime_sales.product_id
left join recent_sales
    on products.product_id = recent_sales.product_id
