import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

CATEGORIES = (
    "Eletrônicos",
    "Casa e Decoração",
    "Moda",
    "Esporte",
    "Beleza",
    "Livros",
    "Brinquedos",
    "Alimentos",
)

PRODUCT_TYPES = {
    "Eletrônicos": ("Fone de ouvido", "Carregador", "Caixa de som", "Teclado"),
    "Casa e Decoração": ("Luminária", "Almofada", "Organizador", "Vaso decorativo"),
    "Moda": ("Camiseta", "Jaqueta", "Tênis", "Bolsa"),
    "Esporte": ("Bola", "Garrafa térmica", "Mochila esportiva", "Faixa elástica"),
    "Beleza": ("Hidratante", "Sabonete", "Protetor solar", "Kit de cuidados"),
    "Livros": ("Livro técnico", "Romance", "Guia prático", "Livro ilustrado"),
    "Brinquedos": ("Jogo educativo", "Quebra-cabeça", "Boneco", "Blocos de montar"),
    "Alimentos": ("Café", "Biscoito", "Granola", "Chocolate"),
}

START_DATE = date(2022, 1, 1)
END_DATE = date(2026, 7, 28)
DEFAULT_OUTPUT_PATH = Path("data/raw/products.csv")


def generate_products(
    quantity: int = 100,
    seed: int = 42,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    if quantity <= 0:
        raise ValueError("quantity deve ser maior que zero")

    random_generator = random.Random(seed)
    fake = Faker("pt_BR")
    fake.seed_instance(seed)
    date_range_days = (END_DATE - START_DATE).days

    products = []
    for index in range(1, quantity + 1):
        category = random_generator.choice(CATEGORIES)
        product_type = random_generator.choice(PRODUCT_TYPES[category])
        unit_price_cents = random_generator.randint(1_000, 250_000)
        unit_cost_cents = random_generator.randint(
            unit_price_cents * 35 // 100,
            unit_price_cents * 80 // 100,
        )
        created_at = START_DATE + timedelta(days=random_generator.randint(0, date_range_days))

        products.append(
            {
                "product_id": f"PROD-{index:06d}",
                "product_name": f"{product_type} {fake.bothify(text='MODELO-??##').upper()}",
                "category": category,
                "brand": fake.bothify(text="MARCA-????").upper(),
                "unit_price": unit_price_cents / 100,
                "unit_cost": unit_cost_cents / 100,
                "stock_quantity": random_generator.randint(0, 500),
                "minimum_stock": random_generator.randint(5, 50),
                "supplier": fake.bothify(text="FORNECEDOR-????-###").upper(),
                "created_at": created_at.isoformat(),
                "is_active": bool(random_generator.getrandbits(1)),
            }
        )

    dataframe = pd.DataFrame(products)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False, encoding="utf-8", float_format="%.2f")

    return dataframe


if __name__ == "__main__":
    generate_products()
