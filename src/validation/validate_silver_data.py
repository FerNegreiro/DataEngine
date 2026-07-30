from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd
from pyarrow.lib import ArrowException

from src.extraction.generate_customers import (
    ACQUISITION_CHANNELS,
    BRAZILIAN_LOCATIONS,
    CUSTOMER_SEGMENTS,
    GENDERS,
)
from src.extraction.generate_orders import (
    ORDER_STATUSES,
    PAYMENT_METHODS,
    SALES_CHANNELS,
)
from src.extraction.generate_products import CATEGORIES
from src.transformation.transform_silver_data import SILVER_SCHEMAS

SilverSource = pd.DataFrame | Path | str

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
NULLABLE_COLUMNS = {
    "customers": set(),
    "orders": {"delivery_date"},
    "order_items": set(),
    "products": set(),
}
MONETARY_COLUMNS = {
    "customers": (),
    "orders": ("shipping_cost", "discount_amount", "order_total"),
    "order_items": ("unit_price", "unit_cost", "line_total"),
    "products": ("unit_price", "unit_cost"),
}


def _append_issue(
    issues: list[str],
    dataset_issues: list[str],
    dataset_name: str,
    message: str,
) -> None:
    detail = f"{dataset_name}: {message}"
    if detail not in issues:
        issues.append(detail)
    if message not in dataset_issues:
        dataset_issues.append(message)


def _load_dataset(
    source: SilverSource,
    dataset_name: str,
    errors: list[str],
    dataset_report: dict[str, object],
) -> pd.DataFrame | None:
    if isinstance(source, pd.DataFrame):
        dataset_report["exists"] = True
        return source.copy()

    path = Path(source)
    dataset_report["path"] = str(path)
    if not path.is_file():
        _append_issue(
            errors,
            dataset_report["errors"],
            dataset_name,
            f"arquivo Silver não encontrado: {path}",
        )
        return None

    dataset_report["exists"] = True
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError, ArrowException) as error:
        _append_issue(
            errors,
            dataset_report["errors"],
            dataset_name,
            f"falha ao ler Parquet Silver em {path}: {error}",
        )
        return None


def _validate_schema(
    dataframe: pd.DataFrame,
    dataset_name: str,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    schema = SILVER_SCHEMAS[dataset_name]
    expected_columns = list(schema["columns"])
    actual_columns = list(dataframe.columns)
    missing_columns = [
        column for column in expected_columns if column not in actual_columns
    ]
    unexpected_columns = [
        column for column in actual_columns if column not in expected_columns
    ]

    if missing_columns:
        _append_issue(
            errors,
            dataset_report["errors"],
            dataset_name,
            f"colunas obrigatórias ausentes: {', '.join(missing_columns)}",
        )
    if unexpected_columns:
        _append_issue(
            errors,
            dataset_report["errors"],
            dataset_name,
            f"colunas não previstas no schema: {', '.join(unexpected_columns)}",
        )

    type_groups = (
        ("strings", pd.api.types.is_string_dtype, "string"),
        ("datetimes", pd.api.types.is_datetime64_any_dtype, "datetime"),
        ("floats", pd.api.types.is_float_dtype, "float64"),
        ("integers", pd.api.types.is_integer_dtype, "inteiro"),
        ("booleans", pd.api.types.is_bool_dtype, "booleano"),
    )
    for group_name, type_check, expected_type in type_groups:
        for column in schema[group_name]:
            if column in dataframe and not type_check(dataframe[column]):
                _append_issue(
                    errors,
                    dataset_report["errors"],
                    dataset_name,
                    f"{column} deve possuir tipo {expected_type}",
                )


def _validate_required_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    nullable = NULLABLE_COLUMNS[dataset_name]
    for column in SILVER_SCHEMAS[dataset_name]["columns"]:
        if column not in dataframe or column in nullable:
            continue
        missing_count = int(dataframe[column].isna().sum())
        if missing_count:
            _append_issue(
                errors,
                dataset_report["errors"],
                dataset_name,
                f"{column} possui {missing_count} valor(es) nulo(s)",
            )


def _validate_primary_key(
    dataframe: pd.DataFrame,
    dataset_name: str,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    primary_key = str(SILVER_SCHEMAS[dataset_name]["primary_key"])
    if primary_key not in dataframe:
        return

    if dataframe[primary_key].isna().any():
        _append_issue(
            errors,
            dataset_report["errors"],
            dataset_name,
            f"chave primária {primary_key} possui valores nulos",
        )
    if dataframe[primary_key].dropna().duplicated().any():
        _append_issue(
            errors,
            dataset_report["errors"],
            dataset_name,
            f"chave primária duplicada em {primary_key}",
        )


def _validate_categories(
    dataframe: pd.DataFrame,
    dataset_name: str,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    allowed_values = {
        "customers": {
            "gender": GENDERS,
            "acquisition_channel": ACQUISITION_CHANNELS,
            "customer_segment": CUSTOMER_SEGMENTS,
        },
        "orders": {
            "order_status": ORDER_STATUSES,
            "payment_method": PAYMENT_METHODS,
            "sales_channel": SALES_CHANNELS,
        },
        "products": {"category": CATEGORIES},
        "order_items": {},
    }
    for column, allowed in allowed_values[dataset_name].items():
        if column in dataframe and (~dataframe[column].dropna().isin(allowed)).any():
            _append_issue(
                errors,
                dataset_report["errors"],
                dataset_name,
                f"{column} possui valores não permitidos",
            )


def _validate_dataset_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    for column in MONETARY_COLUMNS[dataset_name]:
        if column in dataframe:
            values = pd.to_numeric(dataframe[column], errors="coerce")
            if values.isna().sum() > dataframe[column].isna().sum():
                _append_issue(
                    errors,
                    dataset_report["errors"],
                    dataset_name,
                    f"{column} possui valores não numéricos",
                )
            if values.dropna().lt(0).any():
                _append_issue(
                    errors,
                    dataset_report["errors"],
                    dataset_name,
                    f"{column} possui valores monetários negativos",
                )

    if dataset_name == "customers":
        _validate_customer_values(dataframe, errors, dataset_report)
    elif dataset_name == "orders":
        _validate_order_values(dataframe, errors, dataset_report)
    elif dataset_name == "order_items":
        _validate_order_item_values(dataframe, errors, dataset_report)
    elif dataset_name == "products":
        _validate_product_values(dataframe, errors, dataset_report)

    _validate_categories(dataframe, dataset_name, errors, dataset_report)


def _validate_customer_values(
    customers: pd.DataFrame,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    if "email" in customers:
        emails = customers["email"].dropna().astype("string")
        if emails.duplicated().any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "customers",
                "email possui valores duplicados",
            )
        if not emails.str.fullmatch(EMAIL_PATTERN).all():
            _append_issue(
                errors,
                dataset_report["errors"],
                "customers",
                "email possui formato inválido",
            )
        if not emails.eq(emails.str.lower()).all():
            _append_issue(
                errors,
                dataset_report["errors"],
                "customers",
                "email deve estar em letras minúsculas",
            )

    if "state" in customers:
        states = customers["state"].dropna().astype("string")
        if (~states.isin(BRAZILIAN_LOCATIONS)).any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "customers",
                "state possui UFs inválidas",
            )
        if not states.eq(states.str.upper()).all():
            _append_issue(
                errors,
                dataset_report["errors"],
                "customers",
                "state deve estar em letras maiúsculas",
            )

    if {"birth_date", "registration_date"}.issubset(customers.columns):
        birth_dates = pd.to_datetime(customers["birth_date"], errors="coerce")
        registration_dates = pd.to_datetime(
            customers["registration_date"],
            errors="coerce",
        )
        comparable = birth_dates.notna() & registration_dates.notna()
        invalid = 0
        for index in customers.index[comparable]:
            adult_date = _shift_years(birth_dates[index].date(), 18)
            if registration_dates[index].date() < adult_date:
                invalid += 1
        if invalid:
            _append_issue(
                errors,
                dataset_report["errors"],
                "customers",
                "registration_date é anterior ao aniversário de 18 anos",
            )


def _validate_order_values(
    orders: pd.DataFrame,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    if {"order_date", "delivery_date", "order_status"}.issubset(orders.columns):
        order_dates = pd.to_datetime(orders["order_date"], errors="coerce")
        delivery_dates = pd.to_datetime(orders["delivery_date"], errors="coerce")
        delivered = orders["order_status"].eq("Entregue")
        if delivery_dates[delivered].isna().any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "orders",
                "pedidos entregues devem possuir delivery_date",
            )
        comparable = delivery_dates.notna() & order_dates.notna()
        if delivery_dates[comparable].lt(order_dates[comparable]).any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "orders",
                "delivery_date não pode ser anterior a order_date",
            )
        must_be_empty = orders["order_status"].isin({"Processando", "Cancelado"})
        if delivery_dates[must_be_empty].notna().any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "orders",
                "pedidos processando ou cancelados devem ter delivery_date nula",
            )


def _validate_order_item_values(
    order_items: pd.DataFrame,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    if "quantity" in order_items:
        quantities = pd.to_numeric(order_items["quantity"], errors="coerce")
        if quantities.dropna().lt(1).any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "order_items",
                "quantity deve ser maior que zero",
            )
    if "discount_percentage" in order_items:
        discounts = pd.to_numeric(
            order_items["discount_percentage"],
            errors="coerce",
        )
        if (~discounts.dropna().between(0, 100)).any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "order_items",
                "discount_percentage deve estar entre 0 e 100",
            )
    if {"order_id", "product_id"}.issubset(order_items.columns):
        if order_items.duplicated(["order_id", "product_id"]).any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "order_items",
                "produto duplicado dentro do mesmo pedido",
            )


def _validate_product_values(
    products: pd.DataFrame,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    for column in ("stock_quantity", "minimum_stock"):
        if column in products:
            values = pd.to_numeric(products[column], errors="coerce")
            if values.dropna().lt(0).any():
                _append_issue(
                    errors,
                    dataset_report["errors"],
                    "products",
                    f"{column} não pode ser negativo",
                )
    if {"unit_price", "unit_cost"}.issubset(products.columns):
        prices = pd.to_numeric(products["unit_price"], errors="coerce")
        costs = pd.to_numeric(products["unit_cost"], errors="coerce")
        comparable = prices.notna() & costs.notna()
        if costs[comparable].ge(prices[comparable]).any():
            _append_issue(
                errors,
                dataset_report["errors"],
                "products",
                "unit_cost deve ser menor que unit_price",
            )


def _shift_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, month=2, day=28)


def _validate_relationships(
    datasets: Mapping[str, pd.DataFrame],
    errors: list[str],
    dataset_reports: dict[str, dict[str, object]],
) -> None:
    if not {"customers", "orders", "order_items", "products"}.issubset(datasets):
        return

    customers = datasets["customers"]
    orders = datasets["orders"]
    order_items = datasets["order_items"]
    products = datasets["products"]

    if {"customer_id"}.issubset(customers.columns) and {
        "customer_id"
    }.issubset(orders.columns):
        customer_ids = set(customers["customer_id"].dropna())
        if (~orders["customer_id"].dropna().isin(customer_ids)).any():
            _append_issue(
                errors,
                dataset_reports["orders"]["errors"],
                "orders",
                "customer_id possui referências inexistentes em customers",
            )

    if {"order_id"}.issubset(orders.columns) and {
        "order_id"
    }.issubset(order_items.columns):
        order_ids = set(orders["order_id"].dropna())
        if (~order_items["order_id"].dropna().isin(order_ids)).any():
            _append_issue(
                errors,
                dataset_reports["order_items"]["errors"],
                "order_items",
                "order_id possui referências inexistentes em orders",
            )

    if {"product_id"}.issubset(products.columns) and {
        "product_id"
    }.issubset(order_items.columns):
        product_ids = set(products["product_id"].dropna())
        if (~order_items["product_id"].dropna().isin(product_ids)).any():
            _append_issue(
                errors,
                dataset_reports["order_items"]["errors"],
                "order_items",
                "product_id possui referências inexistentes em products",
            )

    _validate_order_dates_against_customers(
        customers,
        orders,
        errors,
        dataset_reports["orders"],
    )
    _validate_financial_reconciliation(
        orders,
        order_items,
        errors,
        dataset_reports["orders"],
    )


def _validate_order_dates_against_customers(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    customer_columns = {"customer_id", "registration_date"}
    order_columns = {"customer_id", "order_date"}
    if not customer_columns.issubset(customers.columns):
        return
    if not order_columns.issubset(orders.columns):
        return

    registrations = customers.drop_duplicates("customer_id").set_index("customer_id")[
        "registration_date"
    ]
    registration_dates = pd.to_datetime(
        orders["customer_id"].map(registrations),
        errors="coerce",
    )
    order_dates = pd.to_datetime(orders["order_date"], errors="coerce")
    comparable = registration_dates.notna() & order_dates.notna()
    if order_dates[comparable].lt(registration_dates[comparable]).any():
        _append_issue(
            errors,
            dataset_report["errors"],
            "orders",
            "order_date é anterior ao cadastro do cliente",
        )


def _validate_financial_reconciliation(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    errors: list[str],
    dataset_report: dict[str, object],
) -> None:
    order_columns = {
        "order_id",
        "shipping_cost",
        "discount_amount",
        "order_total",
    }
    item_columns = {"order_id", "line_total"}
    if not order_columns.issubset(orders.columns):
        return
    if not item_columns.issubset(order_items.columns):
        return

    item_totals = (
        order_items.assign(
            line_total=pd.to_numeric(order_items["line_total"], errors="coerce")
        )
        .groupby("order_id", dropna=False)["line_total"]
        .sum()
    )
    expected_items = pd.to_numeric(orders["order_id"].map(item_totals), errors="coerce")
    if expected_items.isna().any():
        _append_issue(
            errors,
            dataset_report["errors"],
            "orders",
            "existem pedidos sem itens para reconciliação",
        )
        return

    shipping = pd.to_numeric(orders["shipping_cost"], errors="coerce")
    discounts = pd.to_numeric(orders["discount_amount"], errors="coerce")
    actual_totals = pd.to_numeric(orders["order_total"], errors="coerce")
    expected_totals = (expected_items + shipping - discounts).round(2)
    comparable = expected_totals.notna() & actual_totals.notna()
    if expected_totals[comparable].sub(actual_totals[comparable]).abs().gt(0.01).any():
        _append_issue(
            errors,
            dataset_report["errors"],
            "orders",
            "order_total não reconcilia com itens, frete e desconto",
        )


def validate_silver_data(
    sources: Mapping[str, SilverSource],
    input_rows: Mapping[str, int] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_reports: dict[str, dict[str, object]] = {}
    datasets: dict[str, pd.DataFrame] = {}

    for dataset_name in SILVER_SCHEMAS:
        expected_input_rows = (
            None if input_rows is None else input_rows.get(dataset_name)
        )
        dataset_report: dict[str, object] = {
            "exists": False,
            "input_rows": expected_input_rows,
            "output_rows": 0,
            "row_count": 0,
            "columns": [],
            "errors": [],
            "warnings": [],
        }
        dataset_reports[dataset_name] = dataset_report

        if dataset_name not in sources:
            _append_issue(
                errors,
                dataset_report["errors"],
                dataset_name,
                "fonte Silver não informada",
            )
            continue

        dataframe = _load_dataset(
            sources[dataset_name],
            dataset_name,
            errors,
            dataset_report,
        )
        if dataframe is None:
            continue

        datasets[dataset_name] = dataframe
        row_count = len(dataframe)
        dataset_report["output_rows"] = row_count
        dataset_report["row_count"] = row_count
        dataset_report["columns"] = list(dataframe.columns)

        if dataframe.empty:
            _append_issue(
                errors,
                dataset_report["errors"],
                dataset_name,
                "dataset Silver não possui registros",
            )
        if expected_input_rows is not None and row_count != expected_input_rows:
            _append_issue(
                errors,
                dataset_report["errors"],
                dataset_name,
                f"quantidade de linhas mudou de {expected_input_rows} para {row_count}",
            )

        _validate_schema(dataframe, dataset_name, errors, dataset_report)
        _validate_required_values(dataframe, dataset_name, errors, dataset_report)
        _validate_primary_key(dataframe, dataset_name, errors, dataset_report)
        _validate_dataset_values(dataframe, dataset_name, errors, dataset_report)

    _validate_relationships(datasets, errors, dataset_reports)

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "datasets": dataset_reports,
    }
