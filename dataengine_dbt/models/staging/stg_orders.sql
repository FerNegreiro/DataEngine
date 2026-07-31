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
from {{ source('dataengine', 'orders') }}
