with item_totals as (
    select
        order_id,
        round(sum(line_total), 2) as items_net_amount
    from {{ ref('stg_order_items') }}
    group by order_id
)

select
    orders.order_id,
    orders.order_total,
    round(
        item_totals.items_net_amount
        + orders.shipping_cost
        - orders.discount_amount,
        2
    ) as expected_order_total
from {{ ref('stg_orders') }} as orders
left join item_totals
    on orders.order_id = item_totals.order_id
where
    item_totals.order_id is null
    or abs(
        orders.order_total
        - (
            item_totals.items_net_amount
            + orders.shipping_cost
            - orders.discount_amount
        )
    ) > 0.01
