# Trabalho Prático 1 - Predição de receita de filmes com dados do TMDB

Este repositório organiza o trabalho prático 1 em um pipeline reprodutível para coleta de filmes via TMDB, preparação da base, validação estratificada para regressão, comparação sistemática de modelos e análise consolidada dos resultados.

O projeto contempla:

- coleta automatizada de metadados e atributos financeiros de filmes pela API pública do TMDB;
- análise exploratória e pré-processamento da base bruta;
- filtragem dos registros para manter apenas filmes válidos para o problema de regressão;
- codificação de gêneros e idiomas originais em variáveis binárias;
- geração de 10 folds estratificados com base em faixas de `revenue`;
- comparação entre `Dummy Regressor`, `Linear Regression`, `KNN Regressor`, `SVR`, `Decision Tree Regressor`, `Random Forest Regressor`, `Gradient Boosting Regressor` e `XGBoost Regressor`;
- consolidação dos resultados em CSVs com métricas por fold, predições e resumo agregado;
- geração de tabelas e figuras para análise de erros, comportamento por faixa de receita e importância de atributos.

No estado atual salvo no repositório, os principais pontos do pipeline são:

- a coleta do TMDB usa as listas `popular` e `top_rated`, com `pages=500` para cada uma;
- a base bruta persistida em `data/TMDB_movies_original.csv` possui `20000` registros;
- a base processada persistida em `data/TMDB_movies_processed.csv` possui `6918` filmes e `73` colunas;
- a variável-alvo `revenue` é modelada apenas na escala original em dólares;
- a estratificação usa `pd.qcut` com `5` faixas de receita;
- a validação externa usa `StratifiedKFold(n_splits=10, shuffle=True, random_state=222050006)`;
- a busca de hiperparâmetros usa `GridSearchCV` com um holdout interno `80/20` dentro de cada fold de treino;
- as métricas consolidadas são `MSE`, `RMSE`, `MAE` e `R^2`.

---

## Estrutura do projeto

```text
UFSJ_Aprendizado_Maquina_TP1/
├─ code/
│  ├─ tmdb/
│  │  ├─ 01_movies.ipynb                         # Coleta filmes pela API do TMDB
│  │  ├─ 02_analysis_and_preprocessing.ipynb     # EDA, limpeza e engenharia de atributos
│  │  └─ images/                                 # Figuras exportadas na etapa de TMDB
│  └─ revenue/
│     ├─ 01_fold_generation.ipynb                # Gera bins e folds estratificados
│     ├─ 02_model_grid_search.ipynb              # Compara regressores e salva métricas
│     ├─ 03_analysis_no_transform.ipynb          # Análise detalhada do melhor modelo
│     ├─ experiment_utils.py                     # Funções auxiliares compartilhadas
│     └─ images/                                 # Figuras exportadas na etapa de regressão
├─ data/
│  ├─ TMDB_movies_original.csv                   # Base bruta coletada do TMDB
│  ├─ TMDB_movies_processed.csv                  # Base final usada na regressão
│  ├─ revenue_folds.csv                          # Atribuição de fold por id_tmdb
│  ├─ revenue_stratification_bins.csv            # Limites dos bins de revenue
│  └─ revenue_model_selection/
│     ├─ model_selection_results.csv             # Métricas por fold e por modelo
│     ├─ model_selection_predictions.csv         # Predições filme a filme
│     ├─ model_selection_summary.csv             # Resumo médio por modelo
│     └─ error_analysis/
│        ├─ 03_analysis_no_transform_metricas_por_faixa_raw.csv
│        └─ 03_analysis_no_transform_metricas_por_faixa_formatado.csv
├─ LICENSE
├─ README.md
└─ requirements.txt
```

---

## Etapas do pipeline

### 1. Coleta da base bruta

O notebook [code/tmdb/01_movies.ipynb](code/tmdb/01_movies.ipynb) consulta a API v3 do TMDB em duas fontes:

- `popular`
- `top_rated`

Para cada filme encontrado nessas listas, o notebook faz uma segunda consulta ao endpoint de detalhes (`/movie/{id}`) para reunir:

- `id_tmdb`
- `title`
- `original_language`
- `adult`
- `video`
- `genres`
- `status`
- `runtime`
- `belongs_to_collection`
- `budget`
- `revenue`

Ao final, a coleta grava:

- `data/TMDB_movies_original.csv`

O notebook usa explicitamente:

```python
API_KEY = "YOUR_KEY"
BASE_URL = "https://api.themoviedb.org/3"
```

e inclui pausas entre requisições para reduzir problemas de rate limit.

### 2. Análise exploratória e pré-processamento

O notebook [code/tmdb/02_analysis_and_preprocessing.ipynb](code/tmdb/02_analysis_and_preprocessing.ipynb) carrega a base bruta, inspeciona distribuições e aplica as transformações necessárias para a regressão.

As principais regras de filtragem são:

- remoção de duplicatas por `id_tmdb`;
- remoção de linhas totalmente vazias, exceto no campo `belongs_to_collection`;
- manutenção apenas de registros com `budget > 0`;
- manutenção apenas de registros com `revenue > 0`;
- manutenção apenas de registros com `runtime > 0`;
- manutenção apenas de registros com lista de `genres` não vazia;
- exclusão de registros com `video == True`;
- manutenção apenas de filmes com `status == "Released"`.

As principais transformações de atributos são:

- `belongs_to_collection` vira um indicador binário `0/1`;
- `genres` é transformado em colunas binárias via `MultiLabelBinarizer`;
- `original_language` também é codificado em colunas binárias;
- `revenue` permanece como a última coluna do dataset final.

Ao final, o notebook grava:

- `data/TMDB_movies_processed.csv`
- figuras em `code/tmdb/images/`

### 3. Geração dos folds de revenue

O notebook [code/revenue/01_fold_generation.ipynb](code/revenue/01_fold_generation.ipynb) define o protocolo de validação externa do projeto.

O procedimento usado é:

- discretizar `revenue` em `5` faixas com `pd.qcut`;
- usar esses rótulos apenas para estratificação;
- gerar `10` folds com `StratifiedKFold`;
- salvar os folds por `id_tmdb`, e não pela posição da linha.

Os artefatos produzidos são:

- `data/revenue_folds.csv`
- `data/revenue_stratification_bins.csv`
- `code/revenue/images/01_fold_generation_distribuicao_revenue_com_cortes_pd_qcut.png`

### 4. Seleção de modelos e busca de hiperparâmetros

O notebook [code/revenue/02_model_grid_search.ipynb](code/revenue/02_model_grid_search.ipynb) executa a comparação inicial entre os regressores.

Todos os modelos são avaliados com um `Pipeline` contendo:

- `MinMaxScaler`
- o estimador correspondente

Os modelos comparados são:

- `Dummy Regressor`
- `Linear Regression`
- `KNN Regressor`
- `SVR`
- `Decision Tree Regressor`
- `Random Forest Regressor`
- `Gradient Boosting Regressor`
- `XGBoost Regressor`

O protocolo de avaliação é:

- validação externa em `10` folds;
- para cada fold, ajuste de hiperparâmetros com `GridSearchCV`;
- dentro do fold de treino, um holdout interno `80/20` gerado por `train_test_split`;
- scoring principal baseado em `neg_mean_squared_error`;
- consolidação de `MSE`, `RMSE`, `MAE` e `R^2` na escala original em dólares.

Os artefatos produzidos são:

- `data/revenue_model_selection/model_selection_results.csv`
- `data/revenue_model_selection/model_selection_predictions.csv`
- `data/revenue_model_selection/model_selection_summary.csv`

### 5. Análise detalhada do melhor modelo

O notebook [code/revenue/03_analysis_no_transform.ipynb](code/revenue/03_analysis_no_transform.ipynb) carrega os artefatos da seleção de modelos e aprofunda a leitura do melhor candidato na versão sem transformação da variável-alvo.

Essa etapa cobre:

- leitura das métricas médias por modelo;
- seleção do melhor modelo salvo;
- análise de resíduos;
- segmentação do erro por faixas de receita;
- comparação entre valores reais e preditos;
- importância média das features por permutação.

Os artefatos produzidos são:

- `data/revenue_model_selection/error_analysis/03_analysis_no_transform_metricas_por_faixa_raw.csv`
- `data/revenue_model_selection/error_analysis/03_analysis_no_transform_metricas_por_faixa_formatado.csv`
- figuras em `code/revenue/images/`

---

## Como os dados são usados

### Base bruta do TMDB

O arquivo `data/TMDB_movies_original.csv` contém `20000` registros e `11` colunas:

- `id_tmdb`
- `title`
- `original_language`
- `adult`
- `video`
- `genres`
- `status`
- `runtime`
- `belongs_to_collection`
- `budget`
- `revenue`

### Base processada para regressão

O arquivo `data/TMDB_movies_processed.csv` contém `6918` filmes e `73` colunas.

As colunas principais mantidas diretamente são:

- `id_tmdb`
- `title`
- `runtime`
- `adult`
- `belongs_to_collection`
- `budget`
- `revenue`

As demais colunas correspondem a:

- uma codificação binária dos gêneros;
- uma codificação binária do idioma original.

Assim, o problema final de regressão passa a prever `revenue` a partir de atributos numéricos e binários derivados dos metadados dos filmes.

---

## Protocolo de validação e métricas

A validação do projeto foi desenhada para preservar comparabilidade entre modelos e evitar particionamentos arbitrários.

O fluxo atual é:

- `revenue` é discretizada em `5` faixas com `pd.qcut`;
- os bins servem apenas para estratificar os folds;
- a regressão continua sendo feita sobre o valor bruto de `revenue`;
- os folds externos são fixados com `random_state=222050006`;
- os hiperparâmetros são escolhidos com `GridSearchCV` dentro de cada fold de treino;
- as métricas finais são sempre calculadas na escala original em dólares.

Os arquivos consolidados registram:

- `MSE`
- `RMSE`
- `MAE`
- `R^2`

No estado atual dos artefatos salvos em `data/revenue_model_selection/model_selection_summary.csv`, há `8` linhas consolidadas, uma para cada regressor avaliado na versão `Sem transformacao`.

---

## Como executar

### 1. Preparar o ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se o seu ambiente ainda não oferecer suporte a notebooks, abra os arquivos pelo VS Code ou instale manualmente uma interface Jupyter.

### 2. Definir a chave da API do TMDB

Se quiser refazer a coleta do zero, ajuste no notebook `code/tmdb/01_movies.ipynb`:

```python
API_KEY = "YOUR_KEY"
```

Se você pretende apenas reproduzir a parte de modelagem a partir dos CSVs já presentes, essa etapa é opcional.

### 3. Executar os notebooks na ordem correta

Fluxo completo:

```text
code/tmdb/01_movies.ipynb
code/tmdb/02_analysis_and_preprocessing.ipynb
code/revenue/01_fold_generation.ipynb
code/revenue/02_model_grid_search.ipynb
code/revenue/03_analysis_no_transform.ipynb
```

Fluxo mínimo para reproduzir a regressão a partir dos dados já salvos:

```text
code/tmdb/02_analysis_and_preprocessing.ipynb
code/revenue/01_fold_generation.ipynb
code/revenue/02_model_grid_search.ipynb
code/revenue/03_analysis_no_transform.ipynb
```

Se o objetivo for apenas inspecionar os resultados consolidados que já estão versionados no repositório, você pode abrir diretamente:

```text
code/revenue/03_analysis_no_transform.ipynb
```

---

## Saídas geradas

Ao longo do pipeline, o projeto grava:

- `data/TMDB_movies_original.csv`
- `data/TMDB_movies_processed.csv`
- `data/revenue_folds.csv`
- `data/revenue_stratification_bins.csv`
- `data/revenue_model_selection/model_selection_results.csv`
- `data/revenue_model_selection/model_selection_predictions.csv`
- `data/revenue_model_selection/model_selection_summary.csv`
- `data/revenue_model_selection/error_analysis/03_analysis_no_transform_metricas_por_faixa_raw.csv`
- `data/revenue_model_selection/error_analysis/03_analysis_no_transform_metricas_por_faixa_formatado.csv`
- figuras em `code/tmdb/images/`
- figuras em `code/revenue/images/`

---

## Análise em notebook

O notebook [code/revenue/03_analysis_no_transform.ipynb](code/revenue/03_analysis_no_transform.ipynb) é o ponto central para leitura dos resultados finais. Ele permite explorar:

- o ranking médio dos modelos avaliados;
- a distribuição dos resíduos do melhor modelo;
- o comportamento do erro por faixa de receita;
- a dispersão entre valores reais e preditos;
- a importância média das features por permutação.

As tabelas exportadas em `data/revenue_model_selection/error_analysis/` e as figuras salvas em `code/revenue/images/` são derivadas dessa etapa e podem ser reaproveitadas diretamente no relatório do trabalho.
