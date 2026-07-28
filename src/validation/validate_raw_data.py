import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from src.extraction.generate_customers import (
    ACQUISITION_CHANNELS,
    BRAZILIAN_LOCATIONS,
    CUSTOMER_COLUMNS,
    CUSTOMER_SEGMENTS,
    GENDERS,
    REFERENCE_DATE,
    REGISTRATION_START_DATE,
)
from src.extraction.generate_orders import (
    ORDER_COLUMNS,
    ORDER_ITEM_COLUMNS,
    ORDER_START_DATE,
    ORDER_STATUSES,
    PAYMENT_METHODS,
    SALES_CHANNELS,
)
from src.extraction.generate_products import CATEGORIES
from src.extraction.generate_products import START_DATE as PRODUCT_START_DATE

PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "category",
    "brand",
    "unit_price",
    "unit_cost",
    "stock_quantity",
    "minimum_stock",
    "supplier",
    "created_at",
    "is_active",
)

PRODUCTS_PATH = Path("data/raw/products.csv")
CUSTOMERS_PATH = Path("data/raw/customers.csv")
ORDERS_PATH = Path("data/raw/orders.csv")
ORDER_ITEMS_PATH = Path("data/raw/order_items.csv")
MONEY_QUANTUM = Decimal("0.01")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _append_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _load_dataset(
    path: Path | str,
    dataset_name: str,
    required_columns: tuple[str, ...],
    errors: list[str],
) -> pd.DataFrame | None:
    source = Path(path)
    if not source.is_file():
        _append_error(errors, f"Arquivo de {dataset_name} não encontrado: {source}")
        return None

    try:
        dataframe = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        _append_error(errors, f"Arquivo de {dataset_name} está vazio")
        return None
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        _append_error(errors, f"Falha ao ler o arquivo de {dataset_name}: {error}")
        return None

    if dataframe.empty:
        _append_error(errors, f"Arquivo de {dataset_name} não possui registros")

    missing_columns = [column for column in required_columns if column not in dataframe]
    if missing_columns:
        _append_error(
            errors,
            f"Colunas obrigatórias ausentes em {dataset_name}: {', '.join(missing_columns)}",
        )

    return dataframe


def _validate_required_values(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    dataset_name: str,
    errors: list[str],
) -> None:
    for column in columns:
        if column not in dataframe:
            continue
        missing_count = int(dataframe[column].isna().sum())
        if missing_count:
            _append_error(
                errors,
                f"{dataset_name}.{column} possui {missing_count} valor(es) nulo(s)",
            )


def _validate_primary_key(
    dataframe: pd.DataFrame,
    column: str,
    pattern: str,
    dataset_name: str,
    errors: list[str],
) -> None:
    if column not in dataframe:
        return

    values = dataframe[column].dropna().astype(str)
    if values.duplicated().any():
        _append_error(errors, f"Chave primária duplicada em {dataset_name}.{column}")
    if not values.str.fullmatch(pattern).all():
        _append_error(errors, f"Formato inválido em {dataset_name}.{column}")


def _numeric_values(
    dataframe: pd.DataFrame,
    column: str,
    dataset_name: str,
    errors: list[str],
) -> pd.Series | None:
    if column not in dataframe:
        return None

    values = pd.to_numeric(dataframe[column], errors="coerce")
    invalid = dataframe[column].notna() & values.isna()
    if invalid.any():
        _append_error(errors, f"Valores não numéricos em {dataset_name}.{column}")
    return values


def _date_values(
    dataframe: pd.DataFrame,
    column: str,
    dataset_name: str,
    errors: list[str],
) -> pd.Series | None:
    if column not in dataframe:
        return None

    values = pd.to_datetime(dataframe[column], errors="coerce")
    invalid = dataframe[column].notna() & values.isna()
    if invalid.any():
        _append_error(errors, f"Datas inválidas em {dataset_name}.{column}")
    return values


def _boolean_values(
    dataframe: pd.DataFrame,
    column: str,
    dataset_name: str,
    errors: list[str],
) -> pd.Series | None:
    if column not in dataframe:
        return None

    normalized = dataframe[column].map(
        lambda value: str(value).strip().lower() if pd.notna(value) else None
    )
    parsed = normalized.map({"true": True, "false": False})
    invalid = dataframe[column].notna() & parsed.isna()
    if invalid.any():
        _append_error(errors, f"Valores não booleanos em {dataset_name}.{column}")
    return parsed


def _shift_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, month=2, day=28)


def _age_at_reference(birth_date: pd.Timestamp) -> int:
    birth = birth_date.date()
    before_birthday = (REFERENCE_DATE.month, REFERENCE_DATE.day) < (
        birth.month,
        birth.day,
    )
    return REFERENCE_DATE.year - birth.year - before_birthday


def _to_money(value: object) -> Decimal | None:
    try:
        return Decimal(str(value)).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError):
        return None


def _validate_products(products: pd.DataFrame, errors: list[str]) -> None:
    _validate_required_values(products, PRODUCT_COLUMNS, "products", errors)
    _validate_primary_key(
        products,
        "product_id",
        r"PROD-\d{6}",
        "products",
        errors,
    )

    unit_price = _numeric_values(products, "unit_price", "products", errors)
    unit_cost = _numeric_values(products, "unit_cost", "products", errors)
    stock_quantity = _numeric_values(products, "stock_quantity", "products", errors)
    minimum_stock = _numeric_values(products, "minimum_stock", "products", errors)

    if unit_price is not None and unit_price.dropna().le(0).any():
        _append_error(errors, "products.unit_price deve ser maior que zero")
    if unit_cost is not None and unit_cost.dropna().le(0).any():
        _append_error(errors, "products.unit_cost deve ser maior que zero")
    if unit_price is not None and unit_cost is not None:
        comparable = unit_price.notna() & unit_cost.notna()
        if unit_cost[comparable].ge(unit_price[comparable]).any():
            _append_error(errors, "products.unit_cost deve ser menor que unit_price")
    if stock_quantity is not None:
        valid = stock_quantity.dropna()
        if (~valid.between(0, 500) | valid.mod(1).ne(0)).any():
            _append_error(errors, "products.stock_quantity deve ser inteiro entre 0 e 500")
    if minimum_stock is not None:
        valid = minimum_stock.dropna()
        if (~valid.between(5, 50) | valid.mod(1).ne(0)).any():
            _append_error(errors, "products.minimum_stock deve ser inteiro entre 5 e 50")

    if "category" in products:
        invalid_categories = ~products["category"].dropna().isin(CATEGORIES)
        if invalid_categories.any():
            _append_error(errors, "products.category possui categorias não permitidas")

    created_at = _date_values(products, "created_at", "products", errors)
    if created_at is not None:
        valid_dates = created_at.dropna()
        if (
            ~valid_dates.between(
                pd.Timestamp(PRODUCT_START_DATE),
                pd.Timestamp(REFERENCE_DATE),
            )
        ).any():
            _append_error(errors, "products.created_at está fora do intervalo permitido")

    _boolean_values(products, "is_active", "products", errors)


def _validate_customers(
    customers: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:
    _validate_required_values(customers, CUSTOMER_COLUMNS, "customers", errors)
    _validate_primary_key(
        customers,
        "customer_id",
        r"CUST-\d{6}",
        "customers",
        errors,
    )

    if "email" in customers:
        emails = customers["email"].dropna().astype(str)
        if emails.duplicated().any():
            _append_error(errors, "customers.email possui valores duplicados")
        if not emails.str.fullmatch(EMAIL_PATTERN).all():
            _append_error(errors, "customers.email possui formato inválido")

    birth_dates = _date_values(customers, "birth_date", "customers", errors)
    registration_dates = _date_values(
        customers,
        "registration_date",
        "customers",
        errors,
    )
    if birth_dates is not None:
        valid_birth_dates = birth_dates.dropna()
        ages = valid_birth_dates.map(_age_at_reference)
        if not ages.between(18, 80).all():
            _append_error(errors, "customers.birth_date deve representar idade entre 18 e 80")

    if "gender" in customers:
        if (~customers["gender"].dropna().isin(GENDERS)).any():
            _append_error(errors, "customers.gender possui valores não permitidos")

    valid_states = set(BRAZILIAN_LOCATIONS)
    valid_regions = {region for region, _ in BRAZILIAN_LOCATIONS.values()}
    if "state" in customers and (~customers["state"].dropna().isin(valid_states)).any():
        _append_error(errors, "customers.state possui UFs inválidas")
    if "region" in customers and (~customers["region"].dropna().isin(valid_regions)).any():
        _append_error(errors, "customers.region possui regiões inválidas")
    if {"state", "region"}.issubset(customers.columns):
        location_rows = customers.dropna(subset=["state", "region"])
        incoherent = location_rows.apply(
            lambda row: (
                row["state"] in BRAZILIAN_LOCATIONS
                and BRAZILIAN_LOCATIONS[row["state"]][0] != row["region"]
            ),
            axis=1,
        )
        if incoherent.any():
            _append_error(errors, "customers possui incoerência entre UF e região")
    if {"state", "city"}.issubset(customers.columns):
        location_rows = customers.dropna(subset=["state", "city"])
        incoherent = location_rows.apply(
            lambda row: (
                row["state"] in BRAZILIAN_LOCATIONS
                and row["city"] not in BRAZILIAN_LOCATIONS[row["state"]][1]
            ),
            axis=1,
        )
        if incoherent.any():
            _append_error(errors, "customers possui incoerência entre UF e cidade")

    if "acquisition_channel" in customers:
        invalid_channels = ~customers["acquisition_channel"].dropna().isin(
            ACQUISITION_CHANNELS
        )
        if invalid_channels.any():
            _append_error(
                errors,
                "customers.acquisition_channel possui valores não permitidos",
            )
    if "customer_segment" in customers:
        invalid_segments = ~customers["customer_segment"].dropna().isin(CUSTOMER_SEGMENTS)
        if invalid_segments.any():
            _append_error(errors, "customers.customer_segment possui valores não permitidos")

    active_values = _boolean_values(customers, "is_active", "customers", errors)
    if active_values is not None:
        if "customer_segment" in customers:
            inactive = customers["customer_segment"].eq("Inativo")
            if active_values[inactive].eq(True).any():
                _append_error(
                    errors,
                    "Clientes do segmento Inativo devem possuir is_active falso",
                )
        valid_active = active_values.dropna()
        if not valid_active.empty and valid_active.mean() <= 0.5:
            warnings.append("Baixa proporção de clientes ativos")

    if registration_dates is not None:
        valid_registration_dates = registration_dates.dropna()
        if (
            ~valid_registration_dates.between(
                pd.Timestamp(REGISTRATION_START_DATE),
                pd.Timestamp(REFERENCE_DATE),
            )
        ).any():
            _append_error(
                errors,
                "customers.registration_date está fora do intervalo permitido",
            )

    if birth_dates is not None and registration_dates is not None:
        comparable = birth_dates.notna() & registration_dates.notna()
        invalid_registration = 0
        for index in customers.index[comparable]:
            adult_date = _shift_years(birth_dates[index].date(), 18)
            if registration_dates[index].date() < adult_date:
                invalid_registration += 1
        if invalid_registration:
            _append_error(
                errors,
                "customers possui cadastro anterior ao aniversário de 18 anos",
            )


def _validate_orders(
    orders: pd.DataFrame,
    customers: pd.DataFrame | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    required_values = tuple(column for column in ORDER_COLUMNS if column != "delivery_date")
    _validate_required_values(orders, required_values, "orders", errors)
    _validate_primary_key(orders, "order_id", r"ORD-\d{8}", "orders", errors)

    if "customer_id" in orders and customers is not None and "customer_id" in customers:
        customer_ids = set(customers["customer_id"].dropna())
        if (~orders["customer_id"].dropna().isin(customer_ids)).any():
            _append_error(errors, "orders.customer_id possui referências inexistentes")

    order_dates = _date_values(orders, "order_date", "orders", errors)
    if order_dates is not None:
        valid_order_dates = order_dates.dropna()
        if (
            ~valid_order_dates.between(
                pd.Timestamp(ORDER_START_DATE),
                pd.Timestamp(REFERENCE_DATE),
            )
        ).any():
            _append_error(errors, "orders.order_date está fora do intervalo permitido")

    if (
        customers is not None
        and {"customer_id", "registration_date"}.issubset(customers.columns)
        and "customer_id" in orders
        and order_dates is not None
    ):
        customer_registration = customers.drop_duplicates("customer_id").set_index(
            "customer_id"
        )["registration_date"]
        registration_dates = pd.to_datetime(
            orders["customer_id"].map(customer_registration),
            errors="coerce",
        )
        comparable = order_dates.notna() & registration_dates.notna()
        if order_dates[comparable].lt(registration_dates[comparable]).any():
            _append_error(errors, "orders possui pedido anterior ao cadastro do cliente")

    if "order_status" in orders:
        statuses = orders["order_status"].dropna()
        if (~statuses.isin(ORDER_STATUSES)).any():
            _append_error(errors, "orders.order_status possui valores não permitidos")
        if not statuses.empty and statuses.eq("Entregue").mean() <= 0.5:
            warnings.append("Baixa proporção de pedidos entregues")
    if "payment_method" in orders:
        if (~orders["payment_method"].dropna().isin(PAYMENT_METHODS)).any():
            _append_error(errors, "orders.payment_method possui valores não permitidos")
    if "sales_channel" in orders:
        if (~orders["sales_channel"].dropna().isin(SALES_CHANNELS)).any():
            _append_error(errors, "orders.sales_channel possui valores não permitidos")

    shipping_cost = _numeric_values(orders, "shipping_cost", "orders", errors)
    discount_amount = _numeric_values(orders, "discount_amount", "orders", errors)
    order_total = _numeric_values(orders, "order_total", "orders", errors)
    if shipping_cost is not None and shipping_cost.dropna().lt(0).any():
        _append_error(errors, "orders.shipping_cost não pode ser negativo")
    if discount_amount is not None and discount_amount.dropna().lt(0).any():
        _append_error(errors, "orders.discount_amount não pode ser negativo")
    if order_total is not None and "order_status" in orders:
        non_cancelled = orders["order_status"].ne("Cancelado") & order_total.notna()
        if order_total[non_cancelled].le(0).any():
            _append_error(errors, "orders.order_total deve ser positivo")

    if {"order_status", "delivery_date"}.issubset(orders.columns):
        delivery_dates = _date_values(orders, "delivery_date", "orders", errors)
        if delivery_dates is not None:
            delivered = orders["order_status"].eq("Entregue")
            if delivery_dates[delivered].isna().any():
                _append_error(errors, "Pedidos entregues devem possuir delivery_date")
            if order_dates is not None:
                comparable = delivered & delivery_dates.notna() & order_dates.notna()
                if delivery_dates[comparable].lt(order_dates[comparable]).any():
                    _append_error(
                        errors,
                        "orders.delivery_date não pode ser anterior a order_date",
                    )
                sent = (
                    orders["order_status"].eq("Enviado")
                    & delivery_dates.notna()
                    & order_dates.notna()
                )
                if delivery_dates[sent].lt(order_dates[sent]).any():
                    _append_error(
                        errors,
                        "Pedidos enviados possuem delivery_date inválida",
                    )
            must_be_empty = orders["order_status"].isin({"Processando", "Cancelado"})
            if delivery_dates[must_be_empty].notna().any():
                _append_error(
                    errors,
                    "Pedidos processando ou cancelados devem ter delivery_date vazia",
                )


def _validate_order_items(
    order_items: pd.DataFrame,
    orders: pd.DataFrame | None,
    products: pd.DataFrame | None,
    errors: list[str],
) -> None:
    _validate_required_values(order_items, ORDER_ITEM_COLUMNS, "order_items", errors)
    _validate_primary_key(
        order_items,
        "order_item_id",
        r"ITEM-\d{8}",
        "order_items",
        errors,
    )

    if "order_id" in order_items and orders is not None and "order_id" in orders:
        order_ids = set(orders["order_id"].dropna())
        if (~order_items["order_id"].dropna().isin(order_ids)).any():
            _append_error(errors, "order_items.order_id possui referências inexistentes")
    if "product_id" in order_items and products is not None and "product_id" in products:
        product_ids = set(products["product_id"].dropna())
        if (~order_items["product_id"].dropna().isin(product_ids)).any():
            _append_error(errors, "order_items.product_id possui referências inexistentes")

    quantity = _numeric_values(order_items, "quantity", "order_items", errors)
    unit_price = _numeric_values(order_items, "unit_price", "order_items", errors)
    unit_cost = _numeric_values(order_items, "unit_cost", "order_items", errors)
    discount = _numeric_values(
        order_items,
        "discount_percentage",
        "order_items",
        errors,
    )
    line_total = _numeric_values(order_items, "line_total", "order_items", errors)

    if quantity is not None:
        valid = quantity.dropna()
        if (~valid.between(1, 5) | valid.mod(1).ne(0)).any():
            _append_error(errors, "order_items.quantity deve ser inteiro entre 1 e 5")
    if discount is not None and (~discount.dropna().between(0, 30)).any():
        _append_error(
            errors,
            "order_items.discount_percentage deve estar entre 0 e 30",
        )

    if {"order_id", "product_id"}.issubset(order_items.columns):
        duplicated_products = order_items.duplicated(["order_id", "product_id"])
        if duplicated_products.any():
            _append_error(errors, "Produto repetido dentro do mesmo pedido")

    if products is not None and "product_id" in order_items:
        product_columns = {"product_id", "unit_price", "unit_cost"}
        if product_columns.issubset(products.columns):
            product_reference = products.drop_duplicates("product_id").set_index(
                "product_id"
            )
            expected_prices = pd.to_numeric(
                order_items["product_id"].map(product_reference["unit_price"]),
                errors="coerce",
            )
            expected_costs = pd.to_numeric(
                order_items["product_id"].map(product_reference["unit_cost"]),
                errors="coerce",
            )
            if unit_price is not None:
                comparable = unit_price.notna() & expected_prices.notna()
                if unit_price[comparable].sub(expected_prices[comparable]).abs().gt(0.001).any():
                    _append_error(errors, "Preço do item difere do preço do produto")
            if unit_cost is not None:
                comparable = unit_cost.notna() & expected_costs.notna()
                if unit_cost[comparable].sub(expected_costs[comparable]).abs().gt(0.001).any():
                    _append_error(errors, "Custo do item difere do custo do produto")

    if all(value is not None for value in (quantity, unit_price, discount, line_total)):
        invalid_line_totals = 0
        comparable = (
            quantity.notna()
            & unit_price.notna()
            & discount.notna()
            & line_total.notna()
        )
        for index in order_items.index[comparable]:
            expected = (
                Decimal(str(quantity[index]))
                * Decimal(str(unit_price[index]))
                * (Decimal(100) - Decimal(str(discount[index])))
                / Decimal(100)
            ).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
            actual = _to_money(line_total[index])
            if actual != expected:
                invalid_line_totals += 1
        if invalid_line_totals:
            _append_error(errors, "order_items.line_total possui cálculo incorreto")

    if "order_id" in order_items and orders is not None and "order_id" in orders:
        item_counts = order_items.groupby("order_id").size()
        known_order_ids = set(orders["order_id"].dropna())
        missing_orders = known_order_ids.difference(item_counts.index)
        invalid_counts = item_counts[item_counts.index.isin(known_order_ids)].map(
            lambda count: not 1 <= count <= 5
        )
        if missing_orders or invalid_counts.any():
            _append_error(errors, "Cada pedido deve possuir entre 1 e 5 itens")

    required_order_columns = {
        "order_id",
        "shipping_cost",
        "discount_amount",
        "order_total",
    }
    if (
        orders is not None
        and required_order_columns.issubset(orders.columns)
        and {"order_id", "line_total"}.issubset(order_items.columns)
        and line_total is not None
    ):
        item_totals = (
            pd.DataFrame(
                {
                    "order_id": order_items["order_id"],
                    "line_total": line_total,
                }
            )
            .dropna()
            .groupby("order_id")["line_total"]
            .sum()
        )
        invalid_order_totals = 0
        for _, order in orders.iterrows():
            if order["order_id"] not in item_totals:
                continue
            items_total = Decimal(str(item_totals[order["order_id"]]))
            shipping = _to_money(order["shipping_cost"])
            order_discount = _to_money(order["discount_amount"])
            actual_total = _to_money(order["order_total"])
            if None in (shipping, order_discount, actual_total):
                continue
            expected_total = (items_total + shipping - order_discount).quantize(
                MONEY_QUANTUM,
                rounding=ROUND_HALF_UP,
            )
            if actual_total != expected_total:
                invalid_order_totals += 1
        if invalid_order_totals:
            _append_error(errors, "orders.order_total não reconcilia com os itens")


def validate_raw_data(
    products_path: Path | str = PRODUCTS_PATH,
    customers_path: Path | str = CUSTOMERS_PATH,
    orders_path: Path | str = ORDERS_PATH,
    order_items_path: Path | str = ORDER_ITEMS_PATH,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    products = _load_dataset(products_path, "products", PRODUCT_COLUMNS, errors)
    customers = _load_dataset(customers_path, "customers", CUSTOMER_COLUMNS, errors)
    orders = _load_dataset(orders_path, "orders", ORDER_COLUMNS, errors)
    order_items = _load_dataset(
        order_items_path,
        "order_items",
        ORDER_ITEM_COLUMNS,
        errors,
    )

    if products is not None:
        _validate_products(products, errors)
    if customers is not None:
        _validate_customers(customers, errors, warnings)
    if orders is not None:
        _validate_orders(orders, customers, errors, warnings)
    if order_items is not None:
        _validate_order_items(order_items, orders, products, errors)

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "products_rows": 0 if products is None else len(products),
            "customers_rows": 0 if customers is None else len(customers),
            "orders_rows": 0 if orders is None else len(orders),
            "order_items_rows": 0 if order_items is None else len(order_items),
        },
    }


def main() -> int:
    report = validate_raw_data()
    summary = report["summary"]

    print("Validação dos dados brutos")
    print(f"Produtos: {summary['products_rows']}")
    print(f"Clientes: {summary['customers_rows']}")
    print(f"Pedidos: {summary['orders_rows']}")
    print(f"Itens de pedidos: {summary['order_items_rows']}")

    if report["warnings"]:
        print("Avisos:")
        for warning in report["warnings"]:
            print(f"- {warning}")

    if report["errors"]:
        print("Erros:")
        for error in report["errors"]:
            print(f"- {error}")

    result = "válidos" if report["is_valid"] else "inválidos"
    print(f"Resultado: dados {result}")
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
