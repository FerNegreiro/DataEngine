# Fase 5 — Desenho técnico de Machine Learning

- Projeto: **Previsão de Vendas e Estoque Inteligente**
- Data da análise: **31 de julho de 2026**
- Projeto GCP: `dataengine-fernando-2026`
- Região: `southamerica-east1`

> Este documento contém somente análise exploratória e proposta técnica. Nenhum modelo foi
> treinado, nenhum dataset ou tabela foi criado ou alterado, nenhum arquivo foi enviado à AWS
> e nenhuma mudança foi feita no pipeline produtivo.

## 1. Resumo técnico

### Fatos observados

- O histórico de vendas realizadas vai de **6 de janeiro de 2023 a 28 de julho de 2026**,
  totalizando **1.300 dias corridos**.
- O mart diário contém **789 dias com venda**. Eles **não são consecutivos**: a grade diária
  completa contém **511 dias sem venda** (39,31%) distribuídos em 227 intervalos; o maior
  intervalo contínuo sem venda tem 13 dias.
- Existem **2.000 pedidos**, dos quais **1.846 realizados** e **154 cancelados** (7,70%),
  além de 6.070 itens de pedido e 16.879 unidades realizadas.
- Todos os **100 produtos** possuem alguma venda, porém a demanda produto-dia é fortemente
  intermitente: apenas **5.087 de 130.000 observações produto-dia são positivas**. Portanto,
  **96,09%** das observações corretas na grade completa são zero.
- **75 produtos** têm menos de 52 dias com venda e **37** têm menos de 28 dias com venda.
  Isso limita seriamente modelos independentes por produto.
- A sazonalidade mensal é forte, principalmente em novembro e dezembro, e fevereiro é o mês
  mais fraco. O efeito semanal observado é bem menor e deve ser validado, não presumido.
- Há tendência crescente relevante: a inclinação descritiva é de **0,0265 unidade por dia**
  (aproximadamente 9,67 unidades/dia a mais por ano), com correlação de 0,552 entre tempo e
  unidades diárias. Parte dessa tendência decorre do processo sintético de geração dos dados.
- Estoque é somente um retrato atual: 7 produtos estão no estoque mínimo ou abaixo dele, 3
  tiveram demanda dos últimos 30 dias igual ou superior ao estoque atual e 6 não venderam nos
  últimos 30 dias.

### Recomendação

Adotar como primeira solução um **modelo global supervisionado produto-dia**, treinado com
todos os produtos em conjunto, com calendário, identificação do produto e features causais de
lags e janelas móveis. O primeiro candidato deve ser um
`HistGradientBoostingRegressor` com perda Poisson, por combinar relações não lineares,
previsões não negativas e custo baixo para apenas 130 mil linhas. A previsão diária total deve
ser obtida pela soma das previsões por produto, comparada a um benchmark agregado.

A **baseline oficial** deve ser uma média móvel causal de 28 dias; `y(t-7)`, última observação
e média histórica por dia da semana devem permanecer como controles simples. O horizonte
principal recomendado é **14 dias**, com resultados adicionais para 7 e 30 dias. A métrica
principal deve ser **WAPE**, calculada por horizonte e por fold temporal.

BigQuery ML é adequado como benchmark e alternativa de implantação, especialmente
`ARIMA_PLUS` para a série diária agregada e `BOOSTED_TREE_REGRESSOR` para regressão. Ele não
elimina, porém, o problema de esparsidade dos modelos ARIMA independentes por produto.

## 2. Escopo, fontes e definições

### Fontes consultadas

As consultas foram executadas em modo somente leitura usando ADC, sem DDL ou DML, sobre:

- `dataengine_dbt.fct_sales` — uma linha por item de pedido, incluindo cancelamentos;
- `dataengine_dbt.mart_daily_sales` — uma linha por dia com pelo menos uma venda realizada;
- `dataengine_dbt.dim_products` — estado atual de catálogo, preço e estoque;
- `dataengine_dbt.mart_product_performance` — vendas acumuladas, janela de 30 dias e regra
  atual de estoque;
- `dataengine_dbt.stg_orders` — uma linha por pedido, usada para status e cancelamentos;
- SQL, YAML, testes dbt, README e módulos de geração e pipeline existentes no repositório.

### Definições para o ML

| Conceito | Definição proposta |
|---|---|
| Venda realizada | Item cujo `is_realized_sale` é verdadeiro; pedidos cancelados não entram no alvo. |
| Alvo operacional principal | Unidades realizadas (`quantity`) por produto e dia. |
| Venda diária total | Soma das unidades previstas para todos os produtos ativos no dia. |
| Alvo financeiro secundário | Receita diária; deve ser modelada separadamente ou derivada com premissas explícitas de preço e mix. |
| Dia sem venda | Observação válida com alvo zero na grade diária completa; não é valor ausente. |
| Granularidade principal | Uma linha por `product_id` e `sales_date`. |
| Data de referência | 28 de julho de 2026, maior data de venda realizada. |
| Fuso temporal | Datas do BigQuery conforme o contrato atual; nenhuma conversão adicional foi aplicada. |

O alvo em unidades é preferível para risco de estoque. Receita responde também a preço,
desconto e mix, portanto não deve substituir demanda física.

## 3. Diagnóstico dos dados

### 3.1 Cobertura e granularidade

| Indicador | Valor observado |
|---|---:|
| Data mínima, todos os pedidos | 2023-01-06 |
| Data máxima, todos os pedidos | 2026-07-28 |
| Data mínima, vendas realizadas | 2023-01-06 |
| Data máxima, vendas realizadas | 2026-07-28 |
| Dias corridos no intervalo | 1.300 |
| Dias com venda realizada | 789 |
| Dias sem venda na grade completa | 511 |
| Percentual de dias sem venda | 39,31% |
| Intervalos de dias sem venda | 227 |
| Maior intervalo sem venda | 13 dias, de 2023-02-25 a 2023-03-09 |
| Pedidos totais | 2.000 |
| Pedidos realizados | 1.846 |
| Pedidos cancelados | 154 (7,70%) |
| Itens de pedido | 6.070 |
| Itens de venda realizada | 5.619 |
| Unidades realizadas | 16.879 |
| Produtos no catálogo | 100 |
| Produtos ativos atualmente | 52 |
| Produtos com ao menos uma venda | 100 |

**Conclusão factual:** os 789 registros de `mart_daily_sales` representam apenas dias com
venda e não uma série consecutiva. A série correta para ML contém 1.300 linhas no agregado e
130.000 linhas no nível produto-dia, preenchendo lacunas com zero.

### 3.2 Distribuição diária

| Escopo | Dias | Média de pedidos | Mediana de pedidos | Média de unidades | Mediana de unidades | P90 unidades | P99 unidades | Máximo unidades | Média de receita | Mediana de receita |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Grade completa | 1.300 | 1,42 | 1 | 12,984 | 7 | 35 | 83 | 139 | 13.712,97 | 6.104,03 |
| Somente dias com venda | 789 | 2,34 | 2 | 21,393 | 16 | 45 | 91 | 139 | 22.594,25 | 16.879,19 |

Na grade completa, o desvio-padrão diário é 18,021 unidades. A distribuição é assimétrica,
com muitos zeros e cauda longa. Isso torna RMSE útil para monitorar picos, mas inadequada como
único critério de seleção.

### 3.3 Demanda por produto e histórico insuficiente

| Indicador por produto | Valor |
|---|---:|
| Observações produto-dia | 130.000 |
| Observações produto-dia positivas | 5.087 |
| Observações produto-dia iguais a zero | 124.913 (96,09%) |
| Mínimo de dias com venda | 12 |
| P25 de dias com venda | 24 |
| Mediana de dias com venda | 31 |
| P75 de dias com venda | 50 |
| P90 de dias com venda | 98 |
| Máximo de dias com venda | 412 |
| Produtos com menos de 28 dias com venda | 37 |
| Produtos com menos de 52 dias com venda | 75 |
| Mediana de unidades no histórico | 97 |
| Mediana do intervalo médio entre demandas (ADI) | 39,39 dias |

Pelo critério ADI/CV², os **100 produtos** foram classificados como demanda intermitente. A
mediana de CV² nas quantidades positivas é 0,232; o problema dominante é a baixa frequência,
não a variabilidade do tamanho dos pedidos positivos.

Para esta proposta, um produto é considerado **insuficiente para um modelo individual** quando
possui menos de 52 dias com venda. Esse limite não impede participação no modelo global; ele
apenas impede afirmar que há amostra positiva suficiente para ajustar e validar um modelo
separado com sazonalidade semanal.

Os 75 produtos nessa condição são:

`PROD-000002`, `PROD-000003`, `PROD-000004`, `PROD-000005`, `PROD-000006`,
`PROD-000007`, `PROD-000008`, `PROD-000011`, `PROD-000012`, `PROD-000013`,
`PROD-000014`, `PROD-000015`, `PROD-000017`, `PROD-000019`, `PROD-000021`,
`PROD-000023`, `PROD-000026`, `PROD-000027`, `PROD-000028`, `PROD-000029`,
`PROD-000030`, `PROD-000032`, `PROD-000033`, `PROD-000035`, `PROD-000036`,
`PROD-000037`, `PROD-000039`, `PROD-000040`, `PROD-000041`, `PROD-000042`,
`PROD-000043`, `PROD-000044`, `PROD-000045`, `PROD-000046`, `PROD-000047`,
`PROD-000048`, `PROD-000049`, `PROD-000051`, `PROD-000052`, `PROD-000054`,
`PROD-000055`, `PROD-000056`, `PROD-000058`, `PROD-000059`, `PROD-000060`,
`PROD-000062`, `PROD-000063`, `PROD-000064`, `PROD-000065`, `PROD-000066`,
`PROD-000067`, `PROD-000068`, `PROD-000069`, `PROD-000072`, `PROD-000075`,
`PROD-000076`, `PROD-000077`, `PROD-000078`, `PROD-000079`, `PROD-000080`,
`PROD-000083`, `PROD-000084`, `PROD-000085`, `PROD-000086`, `PROD-000087`,
`PROD-000088`, `PROD-000089`, `PROD-000090`, `PROD-000091`, `PROD-000093`,
`PROD-000094`, `PROD-000095`, `PROD-000096`, `PROD-000098`, `PROD-000099`.

Dentro desse grupo, 37 têm menos de 28 dias positivos e representam o caso mais severo:

`PROD-000002`, `PROD-000004`, `PROD-000005`, `PROD-000006`, `PROD-000007`,
`PROD-000008`, `PROD-000011`, `PROD-000012`, `PROD-000013`, `PROD-000015`,
`PROD-000021`, `PROD-000023`, `PROD-000027`, `PROD-000028`, `PROD-000033`,
`PROD-000035`, `PROD-000037`, `PROD-000039`, `PROD-000041`, `PROD-000042`,
`PROD-000044`, `PROD-000045`, `PROD-000046`, `PROD-000056`, `PROD-000060`,
`PROD-000064`, `PROD-000068`, `PROD-000077`, `PROD-000079`, `PROD-000083`,
`PROD-000087`, `PROD-000088`, `PROD-000089`, `PROD-000090`, `PROD-000093`,
`PROD-000095`, `PROD-000096`.

A demanda também é concentrada: o produto líder representa 10,57% das unidades; os 5
primeiros representam 29,41%, os 10 primeiros 41,57% e os 20 primeiros 55,60%. Essa
concentração exige métricas agregadas e por segmento para que o modelo não pareça bom apenas
por acertar os produtos de maior volume.

| Produto | Categoria | Dias com venda | Pedidos | Unidades | Participação nas unidades |
|---|---|---:|---:|---:|---:|
| `PROD-000020` | Casa e Decoração | 412 | 600 | 1.784 | 10,57% |
| `PROD-000018` | Casa e Decoração | 290 | 373 | 1.121 | 6,64% |
| `PROD-000053` | Beleza | 230 | 283 | 854 | 5,06% |
| `PROD-000025` | Alimentos | 195 | 220 | 641 | 3,80% |
| `PROD-000022` | Esporte | 162 | 185 | 564 | 3,34% |

### 3.4 Sazonalidade semanal

| Dia | Dias no calendário | Dias com venda | Média de unidades/dia | Índice de unidades (média = 100) |
|---|---:|---:|---:|---:|
| Domingo | 186 | 116 | 14,323 | 110,3 |
| Segunda | 186 | 110 | 12,914 | 99,5 |
| Terça | 186 | 103 | 12,382 | 95,4 |
| Quarta | 185 | 113 | 10,849 | 83,6 |
| Quinta | 185 | 112 | 12,886 | 99,3 |
| Sexta | 186 | 119 | 13,570 | 104,5 |
| Sábado | 186 | 116 | 13,952 | 107,5 |

Domingo e sábado aparecem acima da média e quarta-feira abaixo. O gerador de pedidos não
define pesos por dia da semana, portanto esse padrão deve ser tratado como sinal descritivo a
ser revalidado nos folds, não como regra de negócio confirmada.

### 3.5 Sazonalidade mensal e tendência

Para reduzir a confusão entre mês e forte crescimento temporal, foi calculado um índice mensal
dentro de cada ano, usando apenas meses completos, e depois feita a média entre anos.

| Mês | Meses completos | Índice normalizado de unidades | Índice normalizado de receita |
|---|---:|---:|---:|
| Janeiro | 3 | 65,0 | 65,0 |
| Fevereiro | 4 | 44,4 | 48,6 |
| Março | 4 | 84,4 | 77,8 |
| Abril | 4 | 74,7 | 74,8 |
| Maio | 4 | 84,5 | 82,6 |
| Junho | 4 | 79,8 | 79,1 |
| Julho | 3 | 71,6 | 69,2 |
| Agosto | 3 | 98,1 | 97,5 |
| Setembro | 3 | 88,2 | 99,4 |
| Outubro | 3 | 78,7 | 83,0 |
| Novembro | 3 | 202,7 | 200,5 |
| Dezembro | 3 | 272,1 | 268,3 |

**Fato observado:** novembro e dezembro têm o efeito sazonal dominante, fevereiro é o vale e
maio apresenta melhora discreta sobre meses adjacentes. Isso é coerente com os pesos presentes
no gerador de dados, embora o crescimento da base ao longo do tempo ainda influencie os níveis.

**Tendência observada:** a média passou de 2,847 unidades/dia no primeiro trimestre parcial de
2023 para 33,901 no segundo trimestre completo de 2026. Julho de 2026 é parcial até o dia 28 e
atinge 63,786 unidades/dia; não deve ser comparado como trimestre completo. O gerador sorteia
datas somente depois do cadastro do cliente, fazendo mais clientes se tornarem elegíveis ao
longo do período. Logo, o modelo aprenderá também uma tendência produzida pela simulação.

### 3.6 Valores atípicos

Pelo critério IQR aplicado à grade diária completa:

- limite superior de unidades: 47,5; **66 dias** acima do limite;
- limite superior de receita: 50.356,20; **72 dias** acima do limite;
- máximos: 139 unidades em 2026-07-28 e receita de 169.049,81 em 2026-07-27.

Os maiores valores concentram-se no fim da série e em meses de alta sazonalidade. Eles não
devem ser removidos automaticamente: podem representar tendência e sazonalidade válidas. A
decisão deve ocorrer dentro de cada fold, comparando resíduos e contexto temporal, nunca com
um filtro calculado sobre todo o histórico.

### 3.7 Qualidade, ausências e limitações

| Verificação | Resultado |
|---|---:|
| Chaves duplicadas em `fct_sales` | 0 |
| Datas duplicadas em `mart_daily_sales` | 0 |
| Produtos duplicados em `dim_products` | 0 |
| Nulos em campos obrigatórios da fact | 0 |
| Nulos em campos obrigatórios de produto | 0 |
| Faixas inválidas de quantidade/preço/custo | 0 |
| Produtos órfãos na fact | 0 |
| Pedidos órfãos na fact | 0 |
| Diferença de unidades entre fact e mart diário | 0 |
| Itens vendidos antes da data atual de criação do produto | 1.558 (25,67% da fact) |
| Itens realizados de produtos atualmente inativos | 2.144 (38,16% dos itens realizados) |

Os cancelamentos estão completos e coerentes: 154 pedidos cancelados, todos sem data de
entrega. O alvo deve continuar excluindo esses pedidos.

Os dois últimos achados impedem usar `created_at` e `is_active` atuais como features históricas
sem cautela. O gerador de pedidos não restringe produtos por data de criação ou estado ativo.
Além disso, `stock_quantity`, `minimum_stock` e `is_active` são snapshots atuais; não há
histórico de estoque, entradas, rupturas, reposições ou lead time. O estoque é gerado
independentemente e não é decrementado pelas vendas.

| Diagnóstico de estoque e demanda recente | Valor observado |
|---|---:|
| Data de referência | 2026-07-28 |
| Estoque atual total | 24.816 unidades |
| Estoque mínimo total | 2.706 unidades |
| Produtos no mínimo ou abaixo | 7 |
| Unidades vendidas nos últimos 7 dias | 641 |
| Unidades vendidas nos últimos 14 dias | 982 |
| Unidades vendidas nos últimos 30 dias | 1.875 |
| Produtos sem venda nos últimos 7 dias | 20 |
| Produtos sem venda nos últimos 14 dias | 12 |
| Produtos sem venda nos últimos 30 dias | 6 |
| Produtos com demanda de 30 dias pelo menos igual ao estoque | 3 |

## 4. Suficiência dos dados por estratégia

| Estratégia | Evidência | Avaliação |
|---|---|---|
| Modelo diário agregado | 1.300 dias, 789 positivos, tendência e sazonalidade mensal | **Suficiente** para baselines, regressão e modelo estatístico simples; validar mudança de regime. |
| Modelo individual por produto | 96,09% de zeros; 75 produtos com menos de 52 dias positivos | **Não recomendado** como solução geral. Pode ser benchmark somente para os 25 produtos mais frequentes. |
| Modelo global com produto como feature | 130.000 linhas e 5.087 eventos positivos compartilhados entre 100 produtos | **Recomendado**; compartilha padrões de calendário e lags sem exigir um modelo por produto. |
| Horizonte de 7 dias | Curto e apoiado por lags recentes | **Viável**, menor incerteza. |
| Horizonte de 14 dias | Equilíbrio entre operação e estabilidade | **Viável e recomendado como principal**. |
| Horizonte de 30 dias | Necessário para planejamento, mas mais sensível à recursão e regime | **Viável com intervalo e caveat de maior incerteza**. |

## 5. Comparação de abordagens

| Abordagem | Vantagens | Limitações neste conjunto | Decisão |
|---|---|---|---|
| Última observação | Quase sem custo, auditável | Em produto-dia, a última observação costuma ser zero; pode obter MAE artificialmente boa | Baseline obrigatório, não baseline principal |
| Média móvel | Causal, barata, acompanha nível recente | Suaviza picos e depende da janela | **Baseline principal: janela de 28 dias** |
| Média sazonal por dia da semana | Interpretável e captura padrão semanal | Efeito semanal é modesto e não foi definido pelo gerador | Baseline secundário |
| Regressão linear | Muito interpretável e rápida | Relações de lags, produto e sazonalidade são não lineares; pode prever negativos | Challenger simples |
| Random Forest | Não linear, robusta e fácil de usar | Modelo maior, mais lento e fraco para extrapolar tendência | Challenger, não primeira escolha |
| Gradient Boosting | Boa relação entre acurácia, custo e manutenção | Exige validação temporal e tratamento explícito de categóricas | **Candidato principal** |
| XGBoost local | Forte em dados tabulares | Nova dependência e complexidade sem evidência prévia de ganho | Não instalar inicialmente |
| Prophet | Tendência/sazonalidade fáceis de explicar | Dependência pesada; pouco ganho provável com 1.300 pontos e alta esparsidade por produto | Não justificado |
| ARIMA/SARIMA | Adequado para a série diária agregada, interpretável | Um modelo por produto não compartilha informação e sofre com demanda intermitente | Benchmark agregado |
| BigQuery ML `ARIMA_PLUS` | Próximo dos dados, operação simples, intervalos e sazonalidade automáticos | Modelos por ID continuam independentes; custo de auto-ARIMA deve ser controlado | Benchmark agregado e opção warehouse-native |
| BigQuery ML boosted tree | Evita dependência local de XGBoost, integra com BigQuery | Features e backtest precisam ser construídos em SQL; previsão recursiva fica menos simples | Challenger futuro |
| Modelo global supervisionado | Compartilha sinal entre produtos, usa calendário, lags e atributos | Requer prevenção rigorosa de leakage e estratégia multi-step | **Abordagem recomendada** |

BigQuery ML suporta `ARIMA_PLUS_XREG` e múltiplas séries por `TIME_SERIES_ID_COL`, além de
regressores boosted tree e random forest. A documentação oficial confirma que a previsão com
regressores externos exige os valores futuros dessas features. Como muitos lags futuros ainda
não existem no momento da previsão, o pipeline teria de fornecê-los recursivamente ou adotar
features conhecidas antecipadamente. Referências: [ARIMA_PLUS_XREG](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-multivariate-time-series),
[ML.FORECAST](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-forecast)
e [regressão no BigQuery ML](https://docs.cloud.google.com/bigquery/docs/regression-overview).

### Ajuste operacional das opções finalistas

| Opção | Custo e treino | Interpretabilidade | Implantação e manutenção | Produto/portfólio |
|---|---|---|---|---|
| Média móvel causal | Mínimos | Alta | Muito simples, Python ou SQL | Excelente baseline auditável |
| HistGradientBoosting global | Baixos para 130 mil linhas | Média, com importância/permutação | Integra bem aos módulos Python; um artefato global | Mostra feature engineering, backtest e serving sem complexidade excessiva |
| BigQuery ML `ARIMA_PLUS` | Baixos no volume atual; auto-ARIMA amplia bytes processados | Alta no agregado | Warehouse-native, pouca infraestrutura | Excelente benchmark de integração BigQuery |
| BigQuery ML boosted tree | Baixos a moderados | Média | Operação centralizada no BigQuery, features em SQL | Bom challenger, mas duplica parte da lógica de features |
| XGBoost/Prophet locais | Dependências e manutenção adicionais | Média | Mais artefatos e maior superfície operacional | Só agregam valor se houver ganho mensurável |

O volume atual é pequeno; tempo e custo de treinamento não justificam infraestrutura
distribuída, tuning extenso ou modelos de deep learning. A opção global em scikit-learn tem a
melhor combinação de controle de validação, manutenção e demonstração de Engenharia de Dados.

## 6. Abordagem recomendada

### 6.1 Estratégia diária

1. Construir uma grade diária completa entre 2023-01-06 e a data de corte.
2. Agregar vendas realizadas por produto e dia e preencher ausência correta com zero.
3. Treinar um modelo global de demanda diária por produto.
4. Produzir previsões diárias de 1 a 30 dias para produtos ativos no momento do score.
5. Somar previsões por produto para obter a previsão diária total em unidades.
6. Comparar o total com uma baseline agregada e, opcionalmente, com `ARIMA_PLUS`.

O candidato inicial é `HistGradientBoostingRegressor(loss="poisson")`. A perda Poisson é
adequada a contagens não negativas; o algoritmo suporta relações não lineares e categóricas
com codificação controlada. A documentação do scikit-learn confirma suporte a perda Poisson e
features categóricas: [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

### 6.2 Estratégia por produto

Não criar 100 modelos independentes na primeira implementação. Usar `product_id` como feature
categórica no modelo global e agregar atributos de categoria e marca. Os 25 produtos com pelo
menos 52 dias positivos podem receber benchmarks individuais para comparação, mas não um
pipeline produtivo separado neste momento.

Para demanda intermitente, acrescentar como diagnóstico futuro uma baseline Croston/SBA. Ela
não substitui as baselines solicitadas, mas ajuda a verificar se o ganho do modelo global é
real e não apenas consequência da grande quantidade de zeros.

### 6.3 Previsão multi-step

Na primeira versão, usar previsão recursiva de um passo: a previsão de `t+1` alimenta os lags
necessários para `t+2`, e assim sucessivamente até 30 dias. O backtest deve reproduzir
exatamente esse comportamento. Modelos diretos por horizonte podem ser avaliados depois apenas
se a acumulação de erro em 30 dias for material.

## 7. Features e prevenção de vazamento

| Feature | Uso proposto | Regra temporal |
|---|---|---|
| Dia da semana, dia do mês, mês, trimestre, fim de semana | Conhecidos no futuro | Podem usar a própria data prevista |
| `product_id` | Compartilhar aprendizado por produto | Categoria codificada, sem informação do alvo |
| Categoria e marca | Padrões entre produtos semelhantes | Snapshot atual; validar estabilidade |
| Lags 1, 7, 14, 28 e 30 | Memória de curto e médio prazo | `y(t-k)` somente |
| Médias móveis 7, 14 e 30 | Nível recente | Aplicar `shift(1)` antes de `rolling` |
| Desvio-padrão móvel | Volatilidade recente | Janela encerrada em `t-1` |
| Vendas acumuladas | Maturidade/volume histórico | Acumulado até `t-1` |
| Tendência temporal | Mudança de nível | Índice crescente calculado desde a data inicial |
| Preço | Potencial driver de demanda | Somente preço histórico observado até `t-1`; excluir preço atual do backtest inicial |
| Estoque, estoque mínimo, produto ativo | Risco e filtro de score | **Não usar no treino histórico** enquanto não houver snapshots temporais |

Definições causais mínimas:

- `lag_k(t) = y(t-k)`;
- `media_7(t) = média[y(t-7), ..., y(t-1)]`;
- `acumulado(t) = soma[y(data inicial), ..., y(t-1)]`.

O cálculo de features deve ocorrer separadamente dentro de cada fold. Normalizadores,
codificadores e imputações devem ser ajustados apenas com o treino. Não usar status final do
pedido, estoque atual, data futura, média global do conjunto completo ou valor real de dias do
horizonte como feature.

## 8. Validação temporal

### Comparação

| Método | Uso | Avaliação |
|---|---|---|
| Holdout temporal | Teste final intocado | Obrigatório, simples e auditável |
| Walk-forward | Reproduz origens sucessivas de previsão | Obrigatório para comparar horizontes e estabilidade |
| Expanding window | Treino cresce a cada origem | **Preferido**, pois o volume inicial é pequeno e a tendência é relevante |
| Divisão aleatória | Mistura passado e futuro | Proibida para seleção e avaliação |

### Cortes concretos

| Etapa | Período | Dias | Dias com venda | Pedidos | Unidades |
|---|---|---:|---:|---:|---:|
| Treino inicial | 2023-01-06 a 2026-03-30 | 1.180 | 672 | 1.320 | 11.978 |
| Validação — fold 1 | 2026-03-31 a 2026-04-29 | 30 | 28 | 86 | 776 |
| Validação — fold 2 | 2026-04-30 a 2026-05-29 | 30 | 29 | 134 | 1.239 |
| Validação — fold 3 | 2026-05-30 a 2026-06-28 | 30 | 30 | 110 | 1.011 |
| Teste final | 2026-06-29 a 2026-07-28 | 30 | 30 | 196 | 1.875 |

Em cada fold, o treino deve incluir apenas datas anteriores ao início da janela de validação.
O fold seguinte expande o treino. O teste final não pode ser consultado para escolher features,
modelo ou hiperparâmetros. Em cada janela de 30 dias, calcular resultados acumulados nos
horizontes 7, 14 e 30. O uso de amostras igualmente espaçadas após completar a grade é coerente
com validadores temporais como
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

O horizonte principal é **14 dias**: oferece utilidade operacional maior que 7 dias sem a
incerteza acumulada de 30 dias. Resultados de 7 e 30 dias devem ser publicados como curto prazo
e planejamento, respectivamente.

## 9. Métricas

| Métrica | Com zeros | Papel |
|---|---|---|
| MAE | Segura | Erro médio em unidades; fácil interpretação |
| RMSE | Segura | Penaliza erros em picos e produtos de alto volume |
| MAPE | **Insegura** | Indefinida quando o real é zero; não usar como métrica de seleção |
| sMAPE | Calculável com convenção para zero-zero | Secundária; instável quando real e previsão são pequenos |
| WAPE | Segura se a soma real do grupo for positiva | **Métrica principal** |
| MASE | Viável se o denominador sazonal não for zero | Secundária no agregado e em segmentos válidos |

**Métrica principal:** WAPE em unidades no horizonte de 14 dias, calculada por fold e no total.
WAPE também deve ser publicada em 7 e 30 dias, por categoria e por faixas de volume. Como os
10 maiores produtos concentram 41,57% das unidades, publicar também MAE macro por produto para
evitar que o resultado seja dominado pelos líderes.

**Métricas secundárias:** MAE, RMSE, sMAPE com `0/0 = 0`, MASE sazonal com período 7 quando o
denominador existir, e viés assinado `sum(previsão-real) / sum(real)`. MAPE pode aparecer apenas
como não aplicável, nunca como critério, pois 96,09% dos produto-dias têm real zero.

## 10. Regra proposta de risco de estoque

A classificação deve continuar baseada em regra transparente, não em um segundo modelo de ML.
Para produto ativo e horizonte principal `H = 14`:

- `S`: estoque atual;
- `M`: estoque mínimo;
- `D_H`: soma da demanda prevista nos próximos `H` dias, limitada a valores não negativos;
- `P = S - D_H`: estoque projetado ao fim do horizonte;
- `C = H × S / D_H`: cobertura estimada em dias; se `D_H = 0`, cobertura é infinita.

Aplicar as classes na seguinte ordem:

| Classe | Regra objetiva |
|---|---|
| Estoque crítico | `S <= 0` ou `P <= 0` |
| Alto risco | `P > 0` e `P <= M` |
| Atenção | `P > M` e `C < 2H` |
| Possível excesso | `P > M` e (`D_H = 0` ou `C > 3H`) |
| Estoque adequado | `P > M` e `2H <= C <= 3H` |

Produtos inativos devem ser excluídos da recomendação de reposição e auditados separadamente.
As faixas `2H` e `3H` são parâmetros iniciais transparentes, não verdades observadas. Devem
ser recalibradas quando houver lead time, lote mínimo, reposições e custo de ruptura/excesso.

Situação atual, usando apenas demanda passada e não previsão: 7 produtos estão no mínimo ou
abaixo, 3 têm demanda de 30 dias pelo menos igual ao estoque, a mediana de estoque é 238, a
mediana de estoque mínimo é 26 e a mediana de demanda de 30 dias é 12 unidades. Esses números
servem para validar a regra, não para medir sua futura acurácia.

## 11. Arquitetura de código proposta

```text
src/
    ml/
        __init__.py
        data_loader.py
        feature_engineering.py
        train.py
        evaluate.py
        predict.py
        inventory_risk.py

pipelines/
    machine_learning/
        __init__.py
        run_ml_pipeline.py

tests/
    ml/
        test_data_loader.py
        test_feature_engineering.py
        test_evaluate.py
        test_predict.py
        test_inventory_risk.py
```

Responsabilidades:

- `data_loader.py`: consultas somente leitura, localização do BigQuery, schemas e datas de
  corte; nenhuma criação implícita de tabela;
- `feature_engineering.py`: grade produto-dia, zeros válidos, lags, rolling, calendário e
  validações anti-leakage;
- `train.py`: baselines, pipeline do candidato, metadados e serialização versionada;
- `evaluate.py`: folds expanding, métricas por horizonte, produto, categoria e volume;
- `predict.py`: previsão recursiva, limites não negativos e agregação diária;
- `inventory_risk.py`: regra, precedência das classes e cobertura;
- `run_ml_pipeline.py`: execução independente e idempotente após dbt aprovado.

`pipelines/run_pipeline.py` não deve ser alterado agora. Na implementação futura, o fluxo deve
ser orquestrado como três etapas independentes: carga Silver, `dbt run/test` e, somente após
sucesso dos marts, `python -m pipelines.machine_learning.run_ml_pipeline`. Uma integração no
pipeline principal deve ser opt-in e posterior à estabilização do ML.

## 12. Arquitetura futura no BigQuery

Recomenda-se criar um dataset separado `dataengine_ml` na mesma região
`southamerica-east1`. Isso isola permissões, custo, retenção e ciclo de vida de artefatos de ML
dos marts analíticos, além de tornar o lineage mais claro.

| Tabela futura | Grão e conteúdo | Partição e cluster sugeridos |
|---|---|---|
| `dataengine_ml.model_metrics` | Uma linha por execução, modelo, split, horizonte e métrica | Partição por `evaluation_date`; cluster por `model_name`, `horizon`, `metric_name` |
| `dataengine_ml.sales_forecast` | Uma linha por data prevista e alvo agregado | Partição por `forecast_date`; cluster por `model_version`, `horizon` |
| `dataengine_ml.product_demand_forecast` | Uma linha por produto, data prevista e execução | Partição por `forecast_date`; cluster por `product_id`, `model_version` |
| `dataengine_ml.inventory_risk` | Uma linha por produto, data de referência, horizonte e execução | Partição por `as_of_date`; cluster por `risk_class`, `product_id` |

Todas devem possuir `run_id`, `model_version`, `feature_version`, `trained_until`,
`created_at` e data de snapshot. A escrita futura deve ser idempotente por `run_id`, com staging
e `MERGE` ou substituição controlada da partição. Nenhuma dessas estruturas foi criada nesta
tarefa.

Para o modelo Python, o artefato `joblib` deve ser versionado fora do BigQuery, inicialmente em
diretório local ignorado pelo Git e depois em armazenamento de objetos autorizado. Se a opção
BigQuery ML for escolhida, o próprio modelo pode residir em `dataengine_ml`.

## 13. Dependências futuras

| Dependência | Estado atual | Recomendação |
|---|---|---|
| `pandas` | Em `requirements.txt` e instalado | Manter |
| `pyarrow` | Em `requirements.txt` e instalado | Manter |
| `google-cloud-bigquery==3.42.3` | Em `requirements.txt` e instalado | Manter |
| `numpy` | Instalado transitivamente, não declarado | Declarar explicitamente |
| `scikit-learn` | Não instalado | Adicionar para pipelines, baselines e modelo global |
| `joblib` | Não instalado | Adicionar para persistir o modelo Python |
| `xgboost` | Não instalado | Não adicionar inicialmente |
| `prophet` | Não instalado | Não adicionar |
| `statsmodels` | Não instalado | Adicionar somente se SARIMA local superar baselines e BigQuery ML não for usado |

Conjunto mínimo recomendado para a implementação Python: `pandas`, `numpy`,
`scikit-learn`, `google-cloud-bigquery`, `pyarrow` e `joblib`. Nenhuma dependência foi
instalada nesta análise.

## 14. Riscos técnicos

1. **Demanda intermitente:** 96,09% de zeros pode favorecer previsões sempre iguais a zero em
   métricas inadequadas.
2. **Mudança de regime:** os últimos 30 dias têm 1.875 unidades, muito acima de folds anteriores;
   o teste final é deliberadamente difícil e não representa um período estacionário.
3. **Dados sintéticos:** sazonalidade e tendência refletem regras do gerador, inclusive datas
   posteriores ao cadastro do cliente.
4. **Catálogo temporal inconsistente:** 25,67% dos itens são anteriores ao `created_at` atual do
   produto; `created_at` e `is_active` não podem entrar no treino histórico sem correção.
5. **Estoque sem histórico:** não há snapshots, reposições, lead time ou decrementar por vendas;
   a classificação de risco é uma simulação operacional, não uma política completa.
6. **Preço sem exposição diária:** preço histórico existe em dias com venda, mas não há série de
   preço/oferta para dias sem venda; preço atual no backtest causaria vazamento.
7. **Erro multi-step:** previsão recursiva acumula erro, especialmente em 30 dias.
8. **Concentração:** WAPE pode ser dominada pelos produtos líderes; métricas macro são obrigatórias.
9. **Coerência hierárquica:** a soma por produto pode divergir de um benchmark agregado; registrar
   o desvio antes de considerar reconciliação mais sofisticada.

## 15. Plano de implementação em etapas

1. Formalizar o contrato de alvo, data de corte e produtos elegíveis.
2. Implementar carregamento somente leitura e grade produto-dia completa.
3. Criar testes de zeros válidos, granularidade, limites de data e ausência de leakage.
4. Implementar baselines: última observação, `y(t-7)`, média móvel 28 e média por dia da semana.
5. Implementar features causais e o candidato global com perda Poisson.
6. Executar três folds expanding e congelar a configuração vencedora.
7. Avaliar uma única vez no teste final de 30 dias.
8. Implementar previsão de 7, 14 e 30 dias e regra de risco de estoque.
9. Criar, após aprovação, `dataengine_ml` e escritas idempotentes.
10. Integrar a execução como etapa independente posterior ao dbt e adicionar monitoramento de
    dados, erro, viés e cobertura.

## 16. Critérios de aceite para a futura implementação

- Grade de 1.300 dias no agregado e 130.000 produto-dias reproduzida para este snapshot.
- Dias sem venda preenchidos com zero; nulos verdadeiros permanecem distintos.
- Cancelamentos excluídos do alvo e preservados para auditoria.
- Nenhuma feature de `t` usa informações posteriores a `t-1`, exceto calendário conhecido.
- Testes automatizados detectam rolling sem `shift`, uso do teste no treino e datas fora do fold.
- Baselines e candidato avaliados nos mesmos folds e horizontes.
- Candidato supera a média móvel de 28 dias em WAPE de 14 dias em pelo menos 2 dos 3 folds de
  validação, sem piora material e sistemática em 7 e 30 dias.
- WAPE, MAE, RMSE, sMAPE, MASE aplicável e viés publicados por horizonte; MAE macro por produto
  e métricas por categoria também publicadas.
- Previsões são não negativas e cobrem todos os produtos elegíveis e todas as datas do horizonte.
- Soma das previsões por produto é reconciliada ou comparada explicitamente ao agregado.
- Regra de estoque é determinística, mutuamente exclusiva e testada nos cinco estados.
- Escrita futura no BigQuery é idempotente, particionada e rastreável por `run_id`.
- O teste final permanece intocado até o congelamento da configuração.

## 17. Questões para fases posteriores

- Qual é o lead time por produto ou fornecedor?
- Existem lote mínimo, estoque em trânsito e pedidos de compra?
- O estoque representa disponibilidade física, vendável ou contábil?
- Há histórico diário de estoque, preço, promoção e ruptura?
- Produtos inativos devem ter previsão somente para auditoria ou ser excluídos totalmente?
- A meta operacional prioriza custo de excesso, ruptura ou nível de serviço?

Essas questões não impedem a primeira implementação de previsão, mas limitam o uso da regra de
estoque como decisão operacional real.

## 18. Conclusão

Os dados são suficientes para um modelo diário agregado e para um modelo global produto-dia,
mas não para sustentar 100 modelos individuais com a mesma qualidade. A combinação de 1.300
dias, forte tendência, sazonalidade mensal e 96,09% de zeros exige validação expanding e
métricas resistentes a zero. A solução mais proporcional ao volume e ao pipeline atual é:

- baseline causal de média móvel de 28 dias;
- candidato global de Gradient Boosting com perda Poisson;
- horizonte principal de 14 dias, mais 7 e 30 dias;
- WAPE como métrica principal, acompanhada de MAE macro, RMSE, sMAPE, MASE aplicável e viés;
- regra transparente de estoque baseada em estoque atual, estoque mínimo, demanda prevista e
  cobertura;
- dataset futuro `dataengine_ml`, separado dos marts e criado somente na fase de implementação.

Nenhum resultado neste documento é uma métrica de modelo: **nenhum modelo foi treinado nesta
tarefa**.
