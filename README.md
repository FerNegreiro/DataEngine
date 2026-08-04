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

### Iterações de demanda intermitente

A primeira iteração confirmou 130.000 produto-dias, com 96,09% de zeros. O challenger
`HistGradientBoostingRegressor(loss="poisson")` perdeu para a média móvel causal de 28 dias
nos três folds de validação e no teste final de 14 dias, por isso não foi promovido.

A segunda iteração classifica cada produto dentro do período de treinamento do respectivo
fold. Ela usa ADI, definido como períodos observados divididos por dias com demanda positiva,
e CV², o quadrado do coeficiente de variação populacional das demandas positivas. Os limites
ADI `1,32` e CV² `0,49` seguem a categorização de Syntetos, Boylan e Croston em
[On the categorization of demand patterns](https://doi.org/10.1057/palgrave.jors.2601841):

- `smooth`: ADI < 1,32 e CV² < 0,49;
- `intermittent`: ADI >= 1,32 e CV² < 0,49;
- `erratic`: ADI < 1,32 e CV² >= 0,49;
- `lumpy`: ADI >= 1,32 e CV² >= 0,49.

Também foram adicionados Croston clássico, Croston-SBA e TSB. O TSB atualiza a probabilidade
de ocorrência em todos os períodos, conforme a abordagem de Teunter, Syntetos e Babai em
[Intermittent demand: Linking forecasting to inventory obsolescence](https://doi.org/10.1016/j.ejor.2011.05.018).
O modelo hurdle global separa a probabilidade de venda da quantidade condicional positiva e
compara regressões Poisson e `squared_error`. Todas as features adicionais são causais.

Execute a segunda iteração sem substituir os artefatos anteriores:

```bash
python -m pipelines.machine_learning.run_ml_pipeline --experiment iteration_02
```

Na execução de referência, os 100 produtos foram classificados como `intermittent` em todos
os folds. O threshold da ocorrência foi `0,50`, escolhido somente nos três folds de validação.
No horizonte principal, o Croston-SBA apresentou WAPE de 133,70%, 124,34% e 129,64%, contra
165,93%, 143,75% e 176,15% da média móvel 28. No teste final, obteve 106,85% contra 115,12%.

Apesar de vencer os três folds e o teste, o Croston-SBA foi rejeitado porque seu viés final de
`-53,22%` excedeu o limite absoluto pré-fixado de 25%. A promoção também exige ausência de
degradação relativa acima de 10%, não piorar materialmente mais de 50% dos segmentos e passar
as verificações de qualidade. A média móvel de 28 dias permanece como champion do cenário de
risco de estoque.

Os resultados ficam em `artifacts/ml/experiments/iteration_02`, incluindo métricas agregadas,
por produto e segmento, classificação ADI/CV², decisão de promoção, previsões, comparação de
risco e modelos locais aplicáveis. Todo esse diretório permanece ignorado pelo Git.

### Publicação produtiva no BigQuery

O champion produtivo permanece `moving_average_28`, versão `1.0.0`. O Croston-SBA, o
HistGradientBoosting Poisson e os dois modelos hurdle permanecem rejeitados; a etapa de
publicação não reavalia nem altera essa decisão. As previsões oficiais são produzidas para 7,
14 e 30 dias, enquanto o risco de estoque usa o horizonte principal de 14 dias.

O modo local continua sendo o padrão e não grava no BigQuery:

```bash
python -m pipelines.machine_learning.run_ml_pipeline
```

A publicação explícita executa a previsão do champion, valida o pacote, cria ou valida o
dataset `dataengine_ml`, publica as tabelas e confere as contagens por `run_id`:

```bash
python -m pipelines.machine_learning.run_ml_pipeline --publish-bigquery
```

O pacote local completo fica em `artifacts/ml/production`, ignorado pelo Git. Ele pode ser
reenviado sem recalcular as previsões somente quando estiver completo, mantiver o champion
aprovado e contiver a decisão oficial da segunda iteração:

```bash
python -m pipelines.machine_learning.publish_ml_results
```

As cinco tabelas históricas usam o projeto `dataengine-fernando-2026`, a região
`southamerica-east1` e o dataset `dataengine_ml`:

- `sales_forecast`: previsões diárias oficiais do champion por execução e horizonte;
- `inventory_risk`: risco por produto no horizonte de 14 dias;
- `model_metrics`: métricas do champion e histórico dos challengers rejeitados;
- `model_registry`: uma linha por nome e versão avaliada, com exatamente um champion ativo;
- `pipeline_runs`: estado, contagens, duração e erro de cada execução.

A carga é idempotente. Cada dataframe passa por uma tabela staging efêmera e depois por
`MERGE` na tabela histórica. O `WRITE_TRUNCATE` é usado somente nessa staging descartável;
nenhum run anterior é apagado. Previsões e riscos usam chaves que incluem `run_id`, métricas
usam a chave composta de execução, modelo, horizonte, período e métrica, e o registry usa
`model_name + model_version`. O mesmo pacote conserva o mesmo `run_id` determinístico.

A autenticação usa Application Default Credentials (ADC). Nenhum arquivo de credencial é
armazenado no projeto. A publicação interrompe antes da carga se encontrar previsão negativa,
duplicidade, horizonte incompleto, risco inválido, metadados incompatíveis ou um challenger nas
previsões oficiais. Depois do `MERGE`, as contagens, os horizontes, o champion e o registro em
`pipeline_runs` são consultados novamente no BigQuery.

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

## Orquestração manual com Apache Airflow

O DAG `dataengine_full_pipeline` coordena o fluxo completo somente por acionamento manual,
sem agendamento e sem catchup. Ele aceita uma execução ativa por vez e segue esta ordem:

```text
validate_environment
  -> run_data_pipeline
  -> dbt_debug
  -> dbt_run
  -> dbt_test
  -> run_ml_pipeline
  -> publish_ml_results
  -> validate_final_outputs
```

`run_ml_pipeline` consulta as fontes analíticas e gera artefatos locais sem usar o modo de
publicação. Isso mantém `moving_average_28` como champion oficial. A tarefa separada
`publish_ml_results` publica uma única vez o pacote produtivo já validado; sua estratégia de
`MERGE` torna novas tentativas com o mesmo pacote idempotentes. O DAG não transfere
DataFrames, Parquets ou modelos por XCom.

Antes de iniciar o fluxo, `validate_environment` verifica dependências, caminhos, AWS STS,
o bucket S3, Google ADC e os três datasets do BigQuery. Ao final,
`validate_final_outputs` consulta somente leitura as relações de origem, dbt e ML e confere a
execução bem-sucedida mais recente, o champion, os horizontes, as duplicidades e as contagens.

As credenciais permanecem fora do repositório. Os diretórios AWS, Google Cloud e dbt
configurados em `airflow/.env` são montados como somente leitura nos containers, conforme os
exemplos de `airflow/.env.example`.
