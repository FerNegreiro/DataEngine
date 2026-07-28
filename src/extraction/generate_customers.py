import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

BRAZILIAN_LOCATIONS = {
    "AC": ("Norte", ("Rio Branco", "Cruzeiro do Sul")),
    "AL": ("Nordeste", ("Maceió", "Arapiraca")),
    "AP": ("Norte", ("Macapá", "Santana")),
    "AM": ("Norte", ("Manaus", "Parintins")),
    "BA": ("Nordeste", ("Salvador", "Feira de Santana")),
    "CE": ("Nordeste", ("Fortaleza", "Juazeiro do Norte")),
    "DF": ("Centro-Oeste", ("Brasília",)),
    "ES": ("Sudeste", ("Vitória", "Vila Velha")),
    "GO": ("Centro-Oeste", ("Goiânia", "Anápolis")),
    "MA": ("Nordeste", ("São Luís", "Imperatriz")),
    "MT": ("Centro-Oeste", ("Cuiabá", "Rondonópolis")),
    "MS": ("Centro-Oeste", ("Campo Grande", "Dourados")),
    "MG": ("Sudeste", ("Belo Horizonte", "Uberlândia")),
    "PA": ("Norte", ("Belém", "Santarém")),
    "PB": ("Nordeste", ("João Pessoa", "Campina Grande")),
    "PR": ("Sul", ("Curitiba", "Londrina")),
    "PE": ("Nordeste", ("Recife", "Caruaru")),
    "PI": ("Nordeste", ("Teresina", "Parnaíba")),
    "RJ": ("Sudeste", ("Rio de Janeiro", "Niterói")),
    "RN": ("Nordeste", ("Natal", "Mossoró")),
    "RS": ("Sul", ("Porto Alegre", "Caxias do Sul")),
    "RO": ("Norte", ("Porto Velho", "Ji-Paraná")),
    "RR": ("Norte", ("Boa Vista", "Rorainópolis")),
    "SC": ("Sul", ("Florianópolis", "Joinville")),
    "SP": ("Sudeste", ("São Paulo", "Campinas")),
    "SE": ("Nordeste", ("Aracaju", "Itabaiana")),
    "TO": ("Norte", ("Palmas", "Araguaína")),
}

GENDERS = ("Feminino", "Masculino", "Não informado")
ACQUISITION_CHANNELS = (
    "Busca orgânica",
    "Anúncios pagos",
    "Redes sociais",
    "Indicação",
    "E-mail marketing",
    "Marketplace",
)
CUSTOMER_SEGMENTS = ("Novo", "Recorrente", "Premium", "Inativo")
NON_INACTIVE_SEGMENTS = CUSTOMER_SEGMENTS[:-1]
CUSTOMER_COLUMNS = (
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
)

REFERENCE_DATE = date(2026, 7, 28)
REGISTRATION_START_DATE = date(2022, 1, 1)
DEFAULT_OUTPUT_PATH = Path("data/raw/customers.csv")


def _shift_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, month=2, day=28)


def _random_date(random_generator: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=random_generator.randint(0, (end - start).days))


def _generate_segments(quantity: int, random_generator: random.Random) -> list[str]:
    segments = random_generator.choices(
        CUSTOMER_SEGMENTS,
        weights=(35, 35, 15, 15),
        k=quantity,
    )
    maximum_inactive = (quantity - 1) // 2
    inactive_indices = [index for index, segment in enumerate(segments) if segment == "Inativo"]

    for index in inactive_indices[maximum_inactive:]:
        segments[index] = random_generator.choice(NON_INACTIVE_SEGMENTS)

    return segments


def _generate_active_flags(
    segments: list[str],
    random_generator: random.Random,
) -> list[bool]:
    active_flags = [
        segment != "Inativo" and random_generator.random() < 0.85 for segment in segments
    ]
    minimum_active = len(segments) // 2 + 1
    active_count = sum(active_flags)

    if active_count < minimum_active:
        for index, segment in enumerate(segments):
            if segment != "Inativo" and not active_flags[index]:
                active_flags[index] = True
                active_count += 1
                if active_count == minimum_active:
                    break

    return active_flags


def generate_customers(
    quantity: int = 500,
    seed: int = 42,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    if quantity <= 0:
        raise ValueError("quantity deve ser maior que zero")

    random_generator = random.Random(seed)
    fake = Faker("pt_BR")
    fake.seed_instance(seed)
    location_items = tuple(BRAZILIAN_LOCATIONS.items())
    oldest_birth_date = _shift_years(REFERENCE_DATE, -81) + timedelta(days=1)
    youngest_birth_date = _shift_years(REFERENCE_DATE, -18)
    segments = _generate_segments(quantity, random_generator)
    active_flags = _generate_active_flags(segments, random_generator)

    customers: list[dict[str, object]] = []
    for index, (segment, is_active) in enumerate(
        zip(segments, active_flags, strict=True),
        start=1,
    ):
        state, (region, cities) = random_generator.choice(location_items)
        birth_date = _random_date(
            random_generator,
            oldest_birth_date,
            youngest_birth_date,
        )
        adult_date = _shift_years(birth_date, 18)
        registration_start = max(REGISTRATION_START_DATE, adult_date)
        registration_date = _random_date(
            random_generator,
            registration_start,
            REFERENCE_DATE,
        )

        customers.append(
            {
                "customer_id": f"CUST-{index:06d}",
                "full_name": fake.name().strip(),
                "email": f"cliente.{index:06d}@example.com",
                "birth_date": birth_date.isoformat(),
                "gender": random_generator.choices(GENDERS, weights=(49, 49, 2), k=1)[0],
                "city": random_generator.choice(cities),
                "state": state,
                "region": region,
                "registration_date": registration_date.isoformat(),
                "acquisition_channel": random_generator.choice(ACQUISITION_CHANNELS),
                "customer_segment": segment,
                "is_active": is_active,
            }
        )

    dataframe = pd.DataFrame(customers, columns=CUSTOMER_COLUMNS)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False, encoding="utf-8")

    return dataframe


if __name__ == "__main__":
    generate_customers()
