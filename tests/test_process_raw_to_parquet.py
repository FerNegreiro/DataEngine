from pathlib import Path
from shutil import copyfile
from typing import NamedTuple

import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import src.transformation.process_raw_to_parquet as processor_module
from src.extraction.generate_customers import generate_customers
from src.extraction.generate_orders import generate_orders
from src.extraction.generate_products import generate_products
from src.transformation.process_raw_to_parquet import process_raw_to_parquet


class RawFiles(NamedTuple):
    products: Path
    customers: Path
    orders: Path
    order_items: Path


class ProcessedData(NamedTuple):
    report: dict[str, object]
    output_dir: Path
    raw_files: RawFiles


@pytest.fixture(scope="module")
def raw_files(tmp_path_factory: pytest.TempPathFactory) -> RawFiles:
    directory = tmp_path_factory.mktemp("parquet_source")
    products = directory / "products.csv"
    customers = directory / "customers.csv"
    orders = directory / "orders.csv"
    order_items = directory / "order_items.csv"
    generate_products(quantity=30, seed=30, output_path=products)
    generate_customers(quantity=60, seed=30, output_path=customers)
    generate_orders(
        quantity=150,
        seed=30,
        customers_path=customers,
        products_path=products,
        orders_output_path=orders,
        order_items_output_path=order_items,
    )
    return RawFiles(products, customers, orders, order_items)


@pytest.fixture(scope="module")
def processed_data(
    tmp_path_factory: pytest.TempPathFactory,
    raw_files: RawFiles,
) -> ProcessedData:
    output_dir = tmp_path_factory.mktemp("parquet_output")
    report = _process(raw_files, output_dir)
    return ProcessedData(report, output_dir, raw_files)


def _process(raw_files: RawFiles, output_dir: Path) -> dict[str, object]:
    return process_raw_to_parquet(
        products_path=raw_files.products,
        customers_path=raw_files.customers,
        orders_path=raw_files.orders,
        order_items_path=raw_files.order_items,
        output_dir=output_dir,
    )


def _copy_raw_files(directory: Path, source: RawFiles) -> RawFiles:
    directory.mkdir(parents=True, exist_ok=True)
    copied = RawFiles(
        directory / "products.csv",
        directory / "customers.csv",
        directory / "orders.csv",
        directory / "order_items.csv",
    )
    for source_path, destination_path in zip(source, copied, strict=True):
        copyfile(source_path, destination_path)
    return copied


def _read_processed(output_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(output_dir / f"{name}.parquet")
        for name in ("products", "customers", "orders", "order_items")
    }


def test_processing_succeeds(processed_data: ProcessedData) -> None:
    assert processed_data.report["success"] is True


def test_creates_all_four_parquet_files(processed_data: ProcessedData) -> None:
    assert set(processed_data.report["files"]) == {
        "products",
        "customers",
        "orders",
        "order_items",
    }
    assert all(Path(path).is_file() for path in processed_data.report["files"].values())


def test_preserves_row_counts(processed_data: ProcessedData) -> None:
    processed = _read_processed(processed_data.output_dir)
    sources = {
        name: pd.read_csv(path)
        for name, path in zip(
            ("products", "customers", "orders", "order_items"),
            processed_data.raw_files,
            strict=True,
        )
    }

    for name in processed:
        assert len(processed[name]) == len(sources[name])
        assert processed_data.report["rows"][name] == len(sources[name])


def test_preserves_columns(processed_data: ProcessedData) -> None:
    processed = _read_processed(processed_data.output_dir)

    for name, source_path in zip(
        ("products", "customers", "orders", "order_items"),
        processed_data.raw_files,
        strict=True,
    ):
        source_columns = list(pd.read_csv(source_path, nrows=0).columns)
        assert list(processed[name].columns) == source_columns
        assert processed_data.report["columns"][name] == source_columns


def test_uses_expected_string_and_numeric_types(processed_data: ProcessedData) -> None:
    processed = _read_processed(processed_data.output_dir)
    string_columns = {
        "products": ["product_id", "product_name", "category", "brand", "supplier"],
        "customers": [
            "customer_id",
            "full_name",
            "email",
            "gender",
            "city",
            "state",
            "region",
            "acquisition_channel",
            "customer_segment",
        ],
        "orders": [
            "order_id",
            "customer_id",
            "order_status",
            "payment_method",
            "sales_channel",
        ],
        "order_items": ["order_item_id", "order_id", "product_id"],
    }

    for name, columns in string_columns.items():
        for column in columns:
            assert pd.api.types.is_string_dtype(processed[name][column])
            assert not isinstance(processed[name][column].dtype, pd.CategoricalDtype)

    assert processed["products"]["unit_price"].dtype == "float64"
    assert processed["products"]["unit_cost"].dtype == "float64"
    assert processed["products"]["stock_quantity"].dtype == "int64"
    assert processed["products"]["minimum_stock"].dtype == "int64"
    assert processed["orders"]["shipping_cost"].dtype == "float64"
    assert processed["orders"]["discount_amount"].dtype == "float64"
    assert processed["orders"]["order_total"].dtype == "float64"
    assert processed["order_items"]["quantity"].dtype == "int64"
    for column in ("unit_price", "unit_cost", "discount_percentage", "line_total"):
        assert processed["order_items"][column].dtype == "float64"


def test_converts_dates_to_datetime(processed_data: ProcessedData) -> None:
    processed = _read_processed(processed_data.output_dir)
    date_columns = {
        "products": ["created_at"],
        "customers": ["birth_date", "registration_date"],
        "orders": ["order_date", "delivery_date"],
    }

    for name, columns in date_columns.items():
        for column in columns:
            assert pd.api.types.is_datetime64_any_dtype(processed[name][column])


def test_preserves_boolean_types(processed_data: ProcessedData) -> None:
    processed = _read_processed(processed_data.output_dir)

    assert pd.api.types.is_bool_dtype(processed["products"]["is_active"])
    assert pd.api.types.is_bool_dtype(processed["customers"]["is_active"])


def test_preserves_null_delivery_dates(processed_data: ProcessedData) -> None:
    raw_orders = pd.read_csv(processed_data.raw_files.orders)
    processed_orders = pd.read_parquet(processed_data.output_dir / "orders.parquet")

    assert processed_orders["delivery_date"].isna().sum() == raw_orders[
        "delivery_date"
    ].isna().sum()


def test_preserves_monetary_values(processed_data: ProcessedData) -> None:
    processed = _read_processed(processed_data.output_dir)
    raw_products = pd.read_csv(processed_data.raw_files.products)
    raw_orders = pd.read_csv(processed_data.raw_files.orders)
    raw_items = pd.read_csv(processed_data.raw_files.order_items)

    for column in ("unit_price", "unit_cost"):
        assert_series_equal(
            processed["products"][column],
            raw_products[column].astype("float64"),
            check_names=False,
        )
    for column in ("shipping_cost", "discount_amount", "order_total"):
        assert_series_equal(
            processed["orders"][column],
            raw_orders[column].astype("float64"),
            check_names=False,
        )
    for column in ("unit_price", "unit_cost", "discount_percentage", "line_total"):
        assert_series_equal(
            processed["order_items"][column],
            raw_items[column].astype("float64"),
            check_names=False,
        )


def test_preserves_referential_integrity(processed_data: ProcessedData) -> None:
    processed = _read_processed(processed_data.output_dir)

    assert set(processed["orders"]["customer_id"]).issubset(
        set(processed["customers"]["customer_id"])
    )
    assert set(processed["order_items"]["order_id"]).issubset(
        set(processed["orders"]["order_id"])
    )
    assert set(processed["order_items"]["product_id"]).issubset(
        set(processed["products"]["product_id"])
    )


def test_uses_snappy_compression(processed_data: ProcessedData) -> None:
    for path in processed_data.report["files"].values():
        metadata = pq.ParquetFile(path).metadata
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                assert row_group.column(column_index).compression == "SNAPPY"


def test_report_has_exact_structure(processed_data: ProcessedData) -> None:
    report = processed_data.report

    assert list(report) == ["success", "files", "rows", "columns"]
    for section in ("files", "rows", "columns"):
        assert list(report[section]) == [
            "products",
            "customers",
            "orders",
            "order_items",
        ]


def test_safely_replaces_existing_files(
    tmp_path: Path,
    raw_files: RawFiles,
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("products", "customers", "orders", "order_items"):
        (tmp_path / f"{name}.parquet").write_bytes(b"conteudo anterior")

    report = _process(raw_files, tmp_path)

    assert report["success"] is True
    for path in report["files"].values():
        assert pq.ParquetFile(path).metadata.num_rows > 0


def test_processing_is_deterministic(
    tmp_path: Path,
    raw_files: RawFiles,
) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    _process(raw_files, first_directory)
    _process(raw_files, second_directory)

    for name in ("products", "customers", "orders", "order_items"):
        first_path = first_directory / f"{name}.parquet"
        second_path = second_directory / f"{name}.parquet"
        assert first_path.read_bytes() == second_path.read_bytes()
        assert_frame_equal(pd.read_parquet(first_path), pd.read_parquet(second_path))


def test_does_not_modify_raw_files(
    tmp_path: Path,
    raw_files: RawFiles,
) -> None:
    before = {path: path.read_bytes() for path in raw_files}

    _process(raw_files, tmp_path)

    assert all(path.read_bytes() == before[path] for path in raw_files)


def test_fails_when_raw_validation_fails(
    tmp_path: Path,
    raw_files: RawFiles,
) -> None:
    copied = _copy_raw_files(tmp_path / "raw", raw_files)
    products = pd.read_csv(copied.products)
    products.loc[0, "unit_cost"] = products.loc[0, "unit_price"]
    products.to_csv(copied.products, index=False)

    with pytest.raises(ValueError, match="Validação dos dados brutos falhou"):
        _process(copied, tmp_path / "processed")

    assert not (tmp_path / "processed").exists()


def test_fails_when_input_files_do_not_exist(tmp_path: Path) -> None:
    missing = RawFiles(
        tmp_path / "products.csv",
        tmp_path / "customers.csv",
        tmp_path / "orders.csv",
        tmp_path / "order_items.csv",
    )

    with pytest.raises(ValueError, match="não encontrado"):
        _process(missing, tmp_path / "processed")


def test_direct_execution_returns_zero_on_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "success": True,
        "files": {"products": "products.parquet"},
        "rows": {"products": 10},
        "columns": {"products": ["product_id"]},
    }
    monkeypatch.setattr(processor_module, "process_raw_to_parquet", lambda: report)

    assert processor_module.main() == 0
    assert "Processamento para Parquet concluído" in capsys.readouterr().out


def test_direct_execution_returns_one_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_processing() -> dict[str, object]:
        raise ValueError("dados inválidos")

    monkeypatch.setattr(processor_module, "process_raw_to_parquet", fail_processing)

    assert processor_module.main() == 1
    assert "dados inválidos" in capsys.readouterr().out
