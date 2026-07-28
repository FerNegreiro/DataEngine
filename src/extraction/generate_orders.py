import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ORDER_COLUMNS = (
    "order_id",
    "customer_id",
    "order_date",
    "order_status",
    "payment_method",
    "sales_channel",
    "shipping_cost",
    "discount_amount",
    "order_total",
    "delivery_date",
)
ORDER_ITEM_COLUMNS = (
    "order_item_id",
    "order_id",
    "product_id",
    "quantity",
    "unit_price",
    "unit_cost",
    "discount_percentage",
    "line_total",
)

ORDER_STATUSES = ("Entregue", "Enviado", "Processando", "Cancelado")
PAYMENT_METHODS = ("Cartão de crédito", "Pix", "Boleto", "Carteira digital")
SALES_CHANNELS = ("Site", "Aplicativo", "Marketplace")
ITEM_DISCOUNTS = (0, 0, 0, 5, 10, 15, 20, 25, 30)
ORDER_DISCOUNTS = (0, 0, 0, 5, 10, 15)
MONTH_WEIGHTS = {
    2: 0.6,
    5: 1.25,
    11: 2.0,
    12: 2.2,
}

ORDER_START_DATE = date(2023, 1, 1)
REFERENCE_DATE = date(2026, 7, 28)
DEFAULT_CUSTOMERS_PATH = Path("data/raw/customers.csv")
DEFAULT_PRODUCTS_PATH = Path("data/raw/products.csv")
DEFAULT_ORDERS_OUTPUT_PATH = Path("data/raw/orders.csv")
DEFAULT_ORDER_ITEMS_OUTPUT_PATH = Path("data/raw/order_items.csv")
MONEY_QUANTUM = Decimal("0.01")
ONE_HUNDRED = Decimal(100)


def _load_input(
    path: Path | str,
    required_columns: set[str],
    dataset_name: str,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Arquivo de {dataset_name} não encontrado: {source}")

    try:
        dataframe = pd.read_csv(source)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"O arquivo de {dataset_name} está vazio") from error

    if dataframe.empty:
        raise ValueError(f"O arquivo de {dataset_name} deve possuir ao menos um registro")

    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas ausentes no arquivo de {dataset_name}: {missing}")

    return dataframe


def _generate_statuses(quantity: int, random_generator: random.Random) -> list[str]:
    statuses = random_generator.choices(
        ORDER_STATUSES,
        weights=(65, 15, 12, 8),
        k=quantity,
    )
    minimum_delivered = quantity // 2 + 1
    delivered_count = statuses.count("Entregue")

    if delivered_count < minimum_delivered:
        for index, status in enumerate(statuses):
            if status != "Entregue":
                statuses[index] = "Entregue"
                delivered_count += 1
                if delivered_count == minimum_delivered:
                    break

    return statuses


def _seasonal_order_date(
    random_generator: random.Random,
    start: date,
    end: date,
) -> date:
    date_range_days = (end - start).days
    maximum_weight = max(MONTH_WEIGHTS.values())

    while True:
        candidate = start + timedelta(days=random_generator.randint(0, date_range_days))
        candidate_weight = MONTH_WEIGHTS.get(candidate.month, 1.0)
        if random_generator.random() <= candidate_weight / maximum_weight:
            return candidate


def _weighted_product_sample(
    products: list[dict[str, object]],
    weights: list[float],
    quantity: int,
    random_generator: random.Random,
) -> list[dict[str, object]]:
    available_products = products.copy()
    available_weights = weights.copy()
    selected_products = []

    for _ in range(quantity):
        selected_index = random_generator.choices(
            range(len(available_products)),
            weights=available_weights,
            k=1,
        )[0]
        selected_products.append(available_products.pop(selected_index))
        available_weights.pop(selected_index)

    return selected_products


def generate_orders(
    quantity: int = 2000,
    seed: int = 42,
    customers_path: Path | str = DEFAULT_CUSTOMERS_PATH,
    products_path: Path | str = DEFAULT_PRODUCTS_PATH,
    orders_output_path: Path | str = DEFAULT_ORDERS_OUTPUT_PATH,
    order_items_output_path: Path | str = DEFAULT_ORDER_ITEMS_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if quantity <= 0:
        raise ValueError("quantity deve ser maior que zero")

    customers = _load_input(
        customers_path,
        {"customer_id", "registration_date"},
        "clientes",
    )
    products = _load_input(
        products_path,
        {"product_id", "unit_price", "unit_cost"},
        "produtos",
    )

    customers = customers.loc[:, ["customer_id", "registration_date"]].copy()
    customers["registration_date"] = pd.to_datetime(
        customers["registration_date"],
        errors="raise",
    ).dt.date
    customers = customers[customers["registration_date"] <= REFERENCE_DATE]
    if customers.empty:
        raise ValueError("Nenhum cliente está apto a realizar pedidos")

    random_generator = random.Random(seed)
    customer_records = customers.to_dict("records")
    random_generator.shuffle(customer_records)
    customer_pool_size = (
        max(1, int(len(customer_records) * 0.85))
        if len(customer_records) > 1
        else 1
    )
    customer_pool = customer_records[:customer_pool_size]
    customer_weights = [
        random_generator.uniform(0.5, 1.5) for _ in range(customer_pool_size)
    ]

    product_records = products.loc[
        :,
        ["product_id", "unit_price", "unit_cost"],
    ].to_dict("records")
    random_generator.shuffle(product_records)
    product_weights = [
        1 / (rank**0.8) for rank in range(1, len(product_records) + 1)
    ]
    statuses = _generate_statuses(quantity, random_generator)

    orders: list[dict[str, object]] = []
    order_items: list[dict[str, object]] = []
    order_item_number = 1

    for order_number, status in enumerate(statuses, start=1):
        customer = random_generator.choices(
            customer_pool,
            weights=customer_weights,
            k=1,
        )[0]
        order_date_start = max(ORDER_START_DATE, customer["registration_date"])
        order_date = _seasonal_order_date(
            random_generator,
            order_date_start,
            REFERENCE_DATE,
        )
        order_id = f"ORD-{order_number:08d}"
        item_count = random_generator.randint(1, min(5, len(product_records)))
        selected_products = _weighted_product_sample(
            product_records,
            product_weights,
            item_count,
            random_generator,
        )

        line_totals = []
        for product in selected_products:
            item_quantity = random_generator.randint(1, 5)
            discount_percentage = random_generator.choice(ITEM_DISCOUNTS)
            unit_price = float(product["unit_price"])
            unit_cost = float(product["unit_cost"])
            line_total_decimal = (
                Decimal(item_quantity)
                * Decimal(str(unit_price))
                * (ONE_HUNDRED - Decimal(discount_percentage))
                / ONE_HUNDRED
            ).quantize(
                MONEY_QUANTUM,
                rounding=ROUND_HALF_UP,
            )
            line_total = float(line_total_decimal)
            line_totals.append(line_total)
            order_items.append(
                {
                    "order_item_id": f"ITEM-{order_item_number:08d}",
                    "order_id": order_id,
                    "product_id": product["product_id"],
                    "quantity": item_quantity,
                    "unit_price": unit_price,
                    "unit_cost": unit_cost,
                    "discount_percentage": discount_percentage,
                    "line_total": line_total,
                }
            )
            order_item_number += 1

        items_total = sum((Decimal(str(total)) for total in line_totals), Decimal("0.00"))
        shipping_cost_decimal = (
            Decimal("0.00")
            if random_generator.random() < 0.25
            else Decimal(random_generator.randint(500, 5_000)) / ONE_HUNDRED
        )
        order_discount_percentage = random_generator.choice(ORDER_DISCOUNTS)
        discount_amount_decimal = (
            items_total * Decimal(order_discount_percentage) / ONE_HUNDRED
        ).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        order_total_decimal = (
            items_total + shipping_cost_decimal - discount_amount_decimal
        ).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        shipping_cost = float(shipping_cost_decimal)
        discount_amount = float(discount_amount_decimal)
        order_total = float(order_total_decimal)

        delivery_date = None
        if status == "Entregue":
            maximum_delivery_days = min(15, (REFERENCE_DATE - order_date).days)
            delivery_date = order_date + timedelta(
                days=random_generator.randint(0, maximum_delivery_days)
            )

        orders.append(
            {
                "order_id": order_id,
                "customer_id": customer["customer_id"],
                "order_date": order_date.isoformat(),
                "order_status": status,
                "payment_method": random_generator.choice(PAYMENT_METHODS),
                "sales_channel": random_generator.choice(SALES_CHANNELS),
                "shipping_cost": shipping_cost,
                "discount_amount": discount_amount,
                "order_total": order_total,
                "delivery_date": delivery_date.isoformat() if delivery_date else None,
            }
        )

    orders_dataframe = pd.DataFrame(orders, columns=ORDER_COLUMNS)
    order_items_dataframe = pd.DataFrame(order_items, columns=ORDER_ITEM_COLUMNS)
    orders_destination = Path(orders_output_path)
    order_items_destination = Path(order_items_output_path)
    orders_destination.parent.mkdir(parents=True, exist_ok=True)
    order_items_destination.parent.mkdir(parents=True, exist_ok=True)
    orders_dataframe.to_csv(
        orders_destination,
        index=False,
        encoding="utf-8",
        float_format="%.2f",
    )
    order_items_dataframe.to_csv(
        order_items_destination,
        index=False,
        encoding="utf-8",
        float_format="%.2f",
    )

    return orders_dataframe, order_items_dataframe


if __name__ == "__main__":
    generate_orders()
