select
    product_id,
    product_name,
    category,
    brand,
    unit_price,
    unit_cost,
    stock_quantity,
    minimum_stock,
    supplier,
    created_at,
    is_active
from {{ source('dataengine', 'products') }}
