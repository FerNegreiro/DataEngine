select
    customer_id,
    full_name,
    email,
    birth_date,
    gender,
    city,
    state,
    region,
    registration_date,
    acquisition_channel,
    customer_segment,
    is_active
from {{ source('dataengine', 'customers') }}
