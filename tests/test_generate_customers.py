import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.extraction.generate_customers import BRAZILIAN_LOCATIONS, generate_customers

EXPECTED_COLUMNS = [
    "customer_id",
    "full_name",
    "email",
    "birth_date",
    "gender",
    "city",
    "state",
    "region",
    "registration_date",
    "acquisition_channel",
    "customer_segment",
    "is_active",
]

STATE_REGION_MAP = {
    "AC": "Norte",
    "AL": "Nordeste",
    "AP": "Norte",
    "AM": "Norte",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "DF": "Centro-Oeste",
    "ES": "Sudeste",
    "GO": "Centro-Oeste",
    "MA": "Nordeste",
    "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "MG": "Sudeste",
    "PA": "Norte",
    "PB": "Nordeste",
    "PR": "Sul",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RJ": "Sudeste",
    "RN": "Nordeste",
    "RS": "Sul",
    "RO": "Norte",
    "RR": "Norte",
    "SC": "Sul",
    "SP": "Sudeste",
    "SE": "Nordeste",
    "TO": "Norte",
}

ALLOWED_GENDERS = {"Feminino", "Masculino", "Não informado"}
ALLOWED_CHANNELS = {
    "Busca orgânica",
    "Anúncios pagos",
    "Redes sociais",
    "Indicação",
    "E-mail marketing",
    "Marketplace",
}
ALLOWED_SEGMENTS = {"Novo", "Recorrente", "Premium", "Inativo"}
REFERENCE_DATE = date(2026, 7, 28)
REGISTRATION_START_DATE = date(2022, 1, 1)


@pytest.fixture(scope="module")
def customers(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    output_path = tmp_path_factory.mktemp("customers") / "customers.csv"
    return generate_customers(quantity=500, seed=42, output_path=output_path)


def _age_on_reference(birth_date: str) -> int:
    birth = date.fromisoformat(birth_date)
    before_birthday = (REFERENCE_DATE.month, REFERENCE_DATE.day) < (birth.month, birth.day)
    return REFERENCE_DATE.year - birth.year - before_birthday


def _eighteenth_birthday(birth_date: str) -> date:
    birth = date.fromisoformat(birth_date)
    try:
        return birth.replace(year=birth.year + 18)
    except ValueError:
        return birth.replace(year=birth.year + 18, month=2, day=28)


def test_generates_requested_quantity(tmp_path: Path) -> None:
    generated = generate_customers(quantity=37, output_path=tmp_path / "customers.csv")

    assert len(generated) == 37


def test_has_exact_expected_columns(customers: pd.DataFrame) -> None:
    assert list(customers.columns) == EXPECTED_COLUMNS


def test_customer_ids_are_unique(customers: pd.DataFrame) -> None:
    assert customers["customer_id"].is_unique


def test_customer_ids_follow_expected_format(customers: pd.DataFrame) -> None:
    assert customers["customer_id"].str.fullmatch(r"CUST-\d{6}").all()


def test_emails_are_unique_and_valid(customers: pd.DataFrame) -> None:
    email_pattern = re.compile(r"^[a-z0-9.]+@[a-z0-9.-]+\.[a-z]{2,}$")

    assert customers["email"].is_unique
    assert customers["email"].map(email_pattern.fullmatch).notna().all()


def test_generated_data_has_no_null_values(customers: pd.DataFrame) -> None:
    assert not customers.isna().any().any()
    assert customers["full_name"].str.strip().ne("").all()


def test_customer_ages_are_between_18_and_80(customers: pd.DataFrame) -> None:
    ages = customers["birth_date"].map(_age_on_reference)

    assert ages.between(18, 80).all()


def test_genders_are_allowed(customers: pd.DataFrame) -> None:
    assert set(customers["gender"]).issubset(ALLOWED_GENDERS)


def test_states_are_valid(customers: pd.DataFrame) -> None:
    assert set(customers["state"]).issubset(STATE_REGION_MAP)


def test_states_and_regions_are_coherent(customers: pd.DataFrame) -> None:
    expected_regions = customers["state"].map(STATE_REGION_MAP)

    assert customers["region"].eq(expected_regions).all()


def test_cities_and_states_are_coherent(customers: pd.DataFrame) -> None:
    assert all(
        city in BRAZILIAN_LOCATIONS[state][1]
        for city, state in zip(customers["city"], customers["state"], strict=True)
    )


def test_acquisition_channels_are_allowed(customers: pd.DataFrame) -> None:
    assert set(customers["acquisition_channel"]).issubset(ALLOWED_CHANNELS)


def test_customer_segments_are_allowed(customers: pd.DataFrame) -> None:
    assert set(customers["customer_segment"]).issubset(ALLOWED_SEGMENTS)


def test_inactive_segment_customers_are_not_active(customers: pd.DataFrame) -> None:
    inactive_customers = customers[customers["customer_segment"] == "Inativo"]

    assert pd.api.types.is_bool_dtype(customers["is_active"])
    assert not inactive_customers["is_active"].any()


def test_active_customers_are_the_majority(customers: pd.DataFrame) -> None:
    assert customers["is_active"].mean() > 0.5


def test_registration_dates_are_in_allowed_interval(customers: pd.DataFrame) -> None:
    registration_dates = pd.to_datetime(customers["registration_date"])

    assert registration_dates.between("2022-01-01", "2026-07-28").all()


def test_registration_occurs_after_turning_eighteen(customers: pd.DataFrame) -> None:
    assert all(
        date.fromisoformat(registration_date) >= _eighteenth_birthday(birth_date)
        for birth_date, registration_date in zip(
            customers["birth_date"],
            customers["registration_date"],
            strict=True,
        )
    )


def test_same_seed_produces_same_data(tmp_path: Path) -> None:
    first = generate_customers(seed=123, output_path=tmp_path / "first.csv")
    second = generate_customers(seed=123, output_path=tmp_path / "second.csv")

    assert_frame_equal(first, second)


def test_different_seeds_produce_different_data(tmp_path: Path) -> None:
    first = generate_customers(seed=123, output_path=tmp_path / "first.csv")
    second = generate_customers(seed=456, output_path=tmp_path / "second.csv")

    assert not first.equals(second)


def test_creates_csv_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "customers.csv"
    generated = generate_customers(quantity=25, output_path=output_path)
    saved = pd.read_csv(output_path)

    assert output_path.is_file()
    assert "Unnamed: 0" not in saved.columns
    assert_frame_equal(saved, generated)
    output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("quantity", [0, -1])
def test_rejects_non_positive_quantity(quantity: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="maior que zero"):
        generate_customers(quantity=quantity, output_path=tmp_path / "customers.csv")
