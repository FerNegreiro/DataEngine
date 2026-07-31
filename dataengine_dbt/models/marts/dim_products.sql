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
    is_active,
    case
        when stock_quantity <= minimum_stock then 'below_minimum'
        else 'adequate'
    end as stock_status
from {{ ref('stg_products') }}
