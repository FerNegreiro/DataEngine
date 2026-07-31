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

## Camadas Bronze e Silver no AWS S3

O pipeline envia os Parquets processados para a camada Bronze e usa a mesma data UTC para
ler, padronizar, validar e publicar os quatro datasets na camada Silver:

```bash
python -m pipelines.run_pipeline
```

As duas camadas são particionadas pela data da execução:

```text
bronze/{dataset}/year={YYYY}/month={MM}/day={DD}/{filename}
silver/{dataset}/year={YYYY}/month={MM}/day={DD}/{filename}
```

A Bronze preserva os dados processados de origem. A Silver normaliza textos, datas, tipos,
valores monetários e ordenação, valida schemas e relacionamentos e só é enviada ao S3 quando
não há erros. O bucket configurado é `dataengine-fernando-2026`.

## Carga Silver no Google BigQuery

Depois de publicar a Silver, o pipeline carrega o mesmo snapshot Parquet nas tabelas:

```text
dataengine-fernando-2026.dataengine.customers
dataengine-fernando-2026.dataengine.orders
dataengine-fernando-2026.dataengine.order_items
dataengine-fernando-2026.dataengine.products
```

O dataset `dataengine` usa a região `southamerica-east1`. A carga é um full refresh:
cada tabela usa `WRITE_TRUNCATE`, portanto o snapshot Silver atual substitui completamente
seu conteúdo anterior. Bronze, Silver e BigQuery compartilham a mesma data UTC da execução.

O fluxo completo é executado com:

```bash
python -m pipelines.run_pipeline
```

A carga BigQuery também pode ser executada isoladamente para uma partição explícita:

```bash
python -m pipelines.loading.load_silver_to_bigquery \
  --execution-date 2026-07-30T00:00:00+00:00
```

A autenticação local usa Application Default Credentials (ADC), sem arquivos de credenciais
no repositório. O projeto deve existir e o dataset deve estar disponível na região correta;
caso o dataset não exista, o pipeline tenta criá-lo quando as permissões permitirem.

## Modelagem analítica com dbt

O projeto dbt em `dataengine_dbt` transforma as quatro tabelas Silver do BigQuery em modelos
analíticos no dataset `dataengine_dbt`, preservando o lineage completo:

```text
dataengine-fernando-2026.dataengine
  -> staging (views)
  -> intermediate (views)
  -> marts (tables)
  -> testes e documentação
```

A autenticação local usa OAuth por meio do profile `dataengine_dbt` em `~/.dbt/profiles.yml`.
Esse arquivo e as credenciais locais não pertencem ao repositório. O dataset de origem é
`dataengine`, o dataset de destino é `dataengine_dbt` e ambos usam a região
`southamerica-east1`.

Os modelos estão organizados em:

- `models/staging`: fontes declaradas e views `stg_` com seleção explícita de colunas;
- `models/intermediate`: enriquecimento de itens e agregação segura de pedidos;
- `models/marts`: dimensões de clientes e produtos, fact de vendas e marts diário, de
  desempenho de produtos e de métricas de clientes.

Os marts de vendas excluem pedidos cancelados das métricas, mas `fct_sales` preserva todos os
itens e disponibiliza `is_realized_sale`. Frete e desconto do pedido não são rateados entre
itens; os campos de pedido na fact são não aditivos. As métricas temporais usam a maior data
de pedido disponível como referência, evitando resultados variáveis com `CURRENT_DATE`.

Execute os comandos abaixo dentro de `dataengine_dbt`:

```bash
dbt debug
dbt parse
dbt compile
dbt run
dbt test
dbt docs generate
```

A documentação é gerada localmente em `target/`. Nenhuma publicação ou hospedagem é feita
por esse fluxo.

## Previsão de demanda e risco de estoque

O pipeline de Machine Learning consulta somente os marts `fct_sales` e `dim_products`, cria
uma grade diária completa por produto e treina um modelo global de demanda com validação
temporal expanding window. Dias sem venda são mantidos com quantidade zero. O modelo nunca
usa split aleatório nem inclui estoque atual nas features históricas.

```bash
python -m pipelines.machine_learning.run_ml_pipeline
```

Por padrão, são previstos 14 dias para produtos ativos. Também é possível gerar horizontes
de 7 ou 30 dias e escolher outro diretório de artefatos:

```bash
python -m pipelines.machine_learning.run_ml_pipeline \
  --forecast-horizon 30 \
  --artifacts-dir artifacts/ml
```

Para uma execução local sem BigQuery, coloque `fct_sales.parquet` e `dim_products.parquet`
em `data/ml_staging` e use `--skip-bigquery`. A execução salva modelo, metadados, métricas,
features, previsões e classificação de risco em `artifacts/ml`. Esse fluxo não cria datasets,
não grava tabelas no BigQuery e não altera o pipeline de engenharia existente.

## Execução com Docker

O container reproduzível executa o pipeline completo: gera produtos, clientes, pedidos e
itens, valida os dados brutos e cria os arquivos Parquet processados.

```bash
docker compose build
docker compose up
```

Como alternativa para uma execução isolada:

```bash
docker compose run --rm dataengine
```

Os volumes `./data/raw:/app/data/raw` e `./data/processed:/app/data/processed` preservam no
host os quatro CSVs brutos e os quatro Parquets gerados pelo container. A execução do fluxo
completo requer credenciais AWS e Google ADC disponíveis para os respectivos SDKs.
