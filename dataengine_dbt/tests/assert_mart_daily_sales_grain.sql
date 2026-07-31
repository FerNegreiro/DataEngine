select
    sales_date,
    count(*) as row_count
from {{ ref('mart_daily_sales') }}
group by sales_date
having count(*) != 1
