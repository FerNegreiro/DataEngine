# DataEngine

Projeto de Engenharia de Dados.

## Tecnologias planejadas

- Python
- Docker
- AWS
- BigQuery
- dbt
- Apache Airflow
- Machine Learning
- Power BI

## Status

O projeto está em fase inicial.

## Estrutura inicial

A base Python e os testes do projeto foram configurados.

## Geração de dados de produtos

O módulo gera dados simulados de produtos de e-commerce para uso futuro nos pipelines e
análises do projeto.

```bash
python -m src.extraction.generate_products
```

O CSV é gerado em `data/raw/products.csv`. Os dados são determinísticos por seed: a mesma
seed produz os mesmos produtos.

## Geração de dados de clientes

O módulo gera dados de clientes brasileiros simulados para uso futuro nos pipelines e
análises do projeto.

```bash
python -m src.extraction.generate_customers
```

O CSV é gerado em `data/raw/customers.csv`. A geração é determinística: a mesma seed produz
os mesmos clientes.

## Geração de pedidos e itens

O módulo gera uma base transacional simulada de e-commerce a partir de
`data/raw/customers.csv` e `data/raw/products.csv`.

```bash
python -m src.extraction.generate_orders
```

São gerados `data/raw/orders.csv` e `data/raw/order_items.csv`, preservando os
relacionamentos entre clientes, produtos, pedidos e itens. A geração é determinística por
seed e aplica sazonalidade simples, com maior volume em novembro e dezembro, aumento em maio
e menor volume em fevereiro.

## Validação dos dados brutos

O módulo valida estrutura, conteúdo, regras de negócio e relacionamentos dos arquivos
`data/raw/products.csv`, `data/raw/customers.csv`, `data/raw/orders.csv` e
`data/raw/order_items.csv`.

```bash
python -m src.validation.validate_raw_data
```

Uma execução bem-sucedida informa dados válidos e termina com código `0`. Dados inválidos
produzem uma lista acumulada de erros e código `1`. Essa validação ocorre antes das próximas
camadas de processamento do pipeline.

## Processamento para Parquet

O módulo valida e converte `data/raw/products.csv`, `data/raw/customers.csv`,
`data/raw/orders.csv` e `data/raw/order_items.csv` nos arquivos correspondentes em
`data/processed`, preservando linhas, colunas, tipos e relacionamentos.

```bash
python -m src.transformation.process_raw_to_parquet
```

O processamento só começa após a validação dos dados brutos e grava os arquivos Parquet com
compressão Snappy. O formato Parquet é adequado para pipelines por oferecer armazenamento
colunar compacto, leitura eficiente e preservação explícita dos tipos de dados.
