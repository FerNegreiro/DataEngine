with row_counts as (
    select
        'customers' as dataset_name,
        (select count(*) from {{ source('dataengine', 'customers') }})
            as source_row_count,
        (select count(*) from {{ ref('stg_customers') }})
            as staging_row_count

    union all

    select
        'orders',
        (select count(*) from {{ source('dataengine', 'orders') }}),
        (select count(*) from {{ ref('stg_orders') }})

    union all

    select
        'order_items',
        (select count(*) from {{ source('dataengine', 'order_items') }}),
        (select count(*) from {{ ref('stg_order_items') }})

    union all

    select
        'products',
        (select count(*) from {{ source('dataengine', 'products') }}),
        (select count(*) from {{ ref('stg_products') }})
)

select
    dataset_name,
    source_row_count,
    staging_row_count
from row_counts
where source_row_count != staging_row_count
