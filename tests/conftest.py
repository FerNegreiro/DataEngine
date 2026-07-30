from __future__ import annotations

import pandas as pd
import pytest

from src.transformation.transform_silver_data import transform_silver_data


@pytest.fixture
def bronze_dataframes() -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.DataFrame(
            [
                {
                    "customer_id": "CUST-000002",
                    "full_name": "  Bruno   Souza ",
                    "email": " BRUNO@example.com ",
                    "birth_date": "1985-06-10",
                    "gender": "Masculino",
                    "city": " Campinas ",
                    "state": "sp",
                    "region": "Sudeste",
                    "registration_date": "2022-03-01",
                    "acquisition_channel": "Redes sociais",
                    "customer_segment": "Recorrente",
                    "is_active": "true",
                },
                {
                    "customer_id": "CUST-000001",
                    "full_name": " Ana Silva ",
                    "email": "ANA@example.com",
                    "birth_date": "1990-01-15",
                    "gender": "Feminino",
                    "city": "São Paulo",
                    "state": "sp",
                    "region": "Sudeste",
                    "registration_date": "2022-01-01",
                    "acquisition_channel": "Busca orgânica",
                    "customer_segment": "Novo",
                    "is_active": "1",
                },
            ]
        ),
        "orders": pd.DataFrame(
            [
                {
                    "order_id": "ORD-00000002",
                    "customer_id": "CUST-000002",
                    "order_date": "2025-02-01",
                    "order_status": " Processando ",
                    "payment_method": "Pix",
                    "sales_channel": "Aplicativo",
                    "shipping_cost": "0",
                    "discount_amount": "0",
                    "order_total": "45",
                    "delivery_date": None,
                },
                {
                    "order_id": "ORD-00000001",
                    "customer_id": "CUST-000001",
                    "order_date": "2025-01-01",
                    "order_status": "Entregue",
                    "payment_method": "Cartão de crédito",
                    "sales_channel": "Site",
                    "shipping_cost": "10",
                    "discount_amount": "5",
                    "order_total": "205",
                    "delivery_date": "2025-01-05",
                },
            ]
        ),
        "order_items": pd.DataFrame(
            [
                {
                    "order_item_id": "ITEM-00000002",
                    "order_id": "ORD-00000002",
                    "product_id": "PROD-000002",
                    "quantity": "1",
                    "unit_price": "50",
                    "unit_cost": "20",
                    "discount_percentage": "10",
                    "line_total": "45",
                },
                {
                    "order_item_id": "ITEM-00000001",
                    "order_id": "ORD-00000001",
                    "product_id": "PROD-000001",
                    "quantity": "2",
                    "unit_price": "100",
                    "unit_cost": "60",
                    "discount_percentage": "0",
                    "line_total": "200",
                },
            ]
        ),
        "products": pd.DataFrame(
            [
                {
                    "product_id": "PROD-000002",
                    "product_name": " Produto   Dois ",
                    "category": "Livros",
                    "brand": " MARCA-B ",
                    "unit_price": "50",
                    "unit_cost": "20",
                    "stock_quantity": "5",
                    "minimum_stock": "2",
                    "supplier": " FORNECEDOR-B ",
                    "created_at": "2023-01-02",
                    "is_active": "false",
                },
                {
                    "product_id": "PROD-000001",
                    "product_name": " Produto Um ",
                    "category": " Eletrônicos ",
                    "brand": "MARCA-A",
                    "unit_price": "100",
                    "unit_cost": "60",
                    "stock_quantity": "10",
                    "minimum_stock": "3",
                    "supplier": "FORNECEDOR-A",
                    "created_at": "2023-01-01",
                    "is_active": "true",
                },
            ]
        ),
    }


@pytest.fixture
def valid_silver_dataframes(
    bronze_dataframes: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return transform_silver_data(bronze_dataframes)
