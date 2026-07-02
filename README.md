# Trabalho Prático - Predição de receita de filmes com dados do TMDB

Este repositório organiza o trabalho prático em um pipeline reprodutível para coleta de filmes via TMDB, preparação da base, validação estratificada para regressão, comparação sistemática de modelos, extensão do conjunto de atributos e análise consolidada dos resultados.

Embora o nome do diretório e algumas pastas internas mantenham a sigla `TP1` por questões de continuidade do projeto, o conteúdo atual do repositório já engloba a continuação do trabalho e deve ser lido como a versão consolidada do trabalho prático.

O projeto contempla:

- coleta automatizada de metadados e atributos financeiros de filmes pela API pública do TMDB;
- análise exploratória e pré-processamento da base bruta;
- filtragem dos registros para manter apenas filmes válidos para o problema de regressão;
- codificação de gêneros e idiomas originais em variáveis binárias;
- enriquecimento da base com metadados adicionais do TMDB, como período de estreia, diretor, elenco principal e empresas de produção;
- geração de 10 folds estratificados com base em faixas de `revenue`;
- comparação entre `Dummy Regressor`, `Linear Regression`, `KNN Regressor`, `SVR`, `Decision Tree Regressor`, `Random Forest Regressor`, `Gradient Boosting Regressor` e `XGBoost Regressor`;
- consolidação dos resultados em CSVs com métricas por fold, predições e resumo agregado para múltiplas versões da variável-alvo;
- geração de tabelas e figuras para análise de erros, comportamento por faixa de receita e importância de atributos;
- investigação de estratégias adicionais para a assimetria da receita, incluindo losses robustas, modelos por faixa, classificação + regressão e soft routing.

No estado atual salvo no repositório, os principais pontos do pipeline são:

- a coleta do TMDB usa as listas `popular` e `top_rated`, com `pages=500` para cada uma;
- a base bruta persistida em `data/TMDB_movies_original.csv` possui `20000` registros;
- a base processada persistida em `data/TMDB_movies_processed.csv` possui `6918` filmes e `73` colunas;
- a base intermediária com metadados adicionais persistida em `data/TMDB_movies_additional_metadata.csv` possui `6918` filmes e `19` colunas;
- a base estendida persistida em `data/TMDB_movies_processed_tmdb_extended.csv` possui `6918` filmes e `287` colunas;
- a variável-alvo `revenue` é modelada tanto na escala original em dólares quanto na transformação `log1p(revenue)`, com reconversão por `expm1` antes da avaliação final;
- a estratificação usa `pd.qcut` com `5` faixas de receita;
- a validação externa usa `StratifiedKFold(n_splits=10, shuffle=True, random_state=222050006)`;
- a busca de hiperparâmetros usa `GridSearchCV` com um holdout interno `80/20` dentro de cada fold de treino;
- as métricas consolidadas são `MSE`, `RMSE`, `MAE` e `R^2`;
- os experimentos adicionais de assimetria reutilizam a mesma divisão externa e a mesma base estendida gerada a partir do TMDB.

---

## Estrutura do projeto

```text
UFSJ_Aprendizado_Maquina_TP1/
├─ code/
│  ├─ tmdb/
│  │  ├─ 01_movies.ipynb                         # Coleta filmes pela API do TMDB
│  │  ├─ 02_analysis_and_preprocessing.ipynb     # EDA, limpeza e engenharia de atributos
│  │  ├─ 03_additional_metadata_enrichment.ipynb # Coleta incremental dos metadados adicionais do TMDB
│  │  ├─ 04_analysis_and_preprocessing_additional_metadata.ipynb # Analisa e codifica os novos metadados para regressão
│  │  ├─ tmdb_api_utils.py                       # Utilitários de credencial e acesso ao TMDB
│  │  ├─ tmdb_feature_enrichment_utils.py        # Coleta incremental e engenharia de novas features do TMDB
│  │  └─ images/                                 # Figuras exportadas na etapa de TMDB
│  └─ revenue/
│     ├─ 01_fold_generation.ipynb                # Gera bins e folds estratificados
│     ├─ 02_model_grid_search.ipynb              # Compara regressores nas duas versões do alvo
│     ├─ 03_analysis_no_transform.ipynb          # Análise detalhada do melhor modelo sem transformação
│     ├─ 04_analysis_log1p.ipynb                 # Análise detalhada do melhor modelo com log1p
│     ├─ 05_tmdb_extended_no_transform.ipynb     # Reavalia os modelos com a base TMDB estendida, sem log1p
│     ├─ 06_tmdb_extended_robust_losses.ipynb                  # Compara perdas mais robustas para lidar com assimetria
│     ├─ 07_tmdb_extended_oracle_band_models.ipynb             # Regressão por faixas com oracle da faixa real no teste
│     ├─ 08_tmdb_extended_classification_plus_regression.ipynb # Estratégia híbrida em dois estágios com roteamento duro
│     ├─ 09_tmdb_extended_soft_routing.ipynb                   # Estratégia híbrida com combinação ponderada dos especialistas
│     ├─ experiment_utils.py                     # Funções auxiliares compartilhadas
│     ├─ imbalance_experiment_utils.py           # Utilitários para experimentos de assimetria
│     └─ images/                                 # Figuras exportadas na etapa de regressão
├─ data/
│  ├─ TMDB_movies_original.csv                   # Base bruta coletada do TMDB
│  ├─ TMDB_movies_processed.csv                  # Base final usada na regressão
│  ├─ TMDB_movies_additional_metadata.csv        # Cache tabular com metadados adicionais coletados do TMDB
│  ├─ TMDB_movies_processed_tmdb_extended.csv    # Base estendida usada nos experimentos posteriores
│  ├─ revenue_folds.csv                          # Atribuição de fold por id_tmdb
│  ├─ revenue_stratification_bins.csv            # Limites dos bins de revenue
│  ├─ revenue_model_selection/
│  │  ├─ model_selection_results.csv             # Métricas por fold e por modelo
│  │  ├─ model_selection_predictions.csv         # Predições filme a filme
│  │  ├─ model_selection_summary.csv             # Resumo médio por modelo
│  │  └─ error_analysis/
│  ├─ revenue_model_selection_tmdb_extended_no_transform/
│  │  ├─ baseline_comparison_summary.csv         # Comparação entre base original e base TMDB estendida
│  │  └─ error_analysis/
│  ├─ revenue_robust_losses_tmdb_extended_promising/
│  │  ├─ best_robust_vs_global_best.csv          # Comparação entre losses robustas e melhor baseline estendido
│  │  └─ error_analysis/
│  ├─ revenue_oracle_band_models_tmdb_extended/
│  │  ├─ comparison_with_global_models.csv       # Ganho potencial dos modelos por faixa com oracle
│  │  └─ error_analysis/
│  ├─ revenue_hybrid_classification_regression_tmdb_extended/
│  │  ├─ best_hybrid_vs_global_best.csv          # Comparação do pipeline híbrido com o melhor modelo global
│  │  └─ error_analysis/
│  └─ revenue_soft_routing_tmdb_extended/
│     ├─ best_soft_routing_vs_global_and_oracle.csv # Comparação final entre soft routing, global e oracle
│     └─ error_analysis/
├─ documents/
│  ├─ tp1/
│  │  ├─ apresentacao_da_versao_parcial_do_artigo.pdf
│  │  ├─ versao_parcial_artigo.pdf               # Versão parcial do artigo do projeto
│  │  └─ versao_final_artigo.pdf                 # Versão final da primeira etapa do artigo
│  └─ tp2/
│     └─ versao_final_artigo_parte2.pdf          # Versão final consolidada da continuação do artigo
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

O notebook lê a chave do TMDB a partir do arquivo `.env` na raiz do projeto ou da variável de ambiente `TMDB_API_KEY`. A URL base da API permanece:

```python
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

Na etapa de modelagem, essa mesma base processada é reutilizada em duas configurações de alvo: `revenue` em dólares e `log1p(revenue)`. Quando a transformação é usada, as predições são reconvertidas com `expm1` antes do cálculo das métricas finais.

### 3. Enriquecimento adicional com metadados do TMDB

O notebook [code/tmdb/03_additional_metadata_enrichment.ipynb](code/tmdb/03_additional_metadata_enrichment.ipynb) consulta incrementalmente o TMDB para montar o cache bruto com novos atributos ainda disponíveis antes do lançamento do filme ou derivados de metadados estáveis do próprio TMDB.

As novas informações incluem:

- `release_date` e variáveis derivadas de período de estreia;
- diretor principal;
- nomes do elenco principal;
- empresa de produção principal e conjunto de empresas associadas;
- contagens agregadas de elenco e de empresas.

Os artefatos produzidos são:

- `data/TMDB_movies_additional_metadata.csv`

### 3.1. Análise e pré-processamento dos metadados adicionais

O notebook [code/tmdb/04_analysis_and_preprocessing_additional_metadata.ipynb](code/tmdb/04_analysis_and_preprocessing_additional_metadata.ipynb) exerce para esses novos atributos o mesmo papel que o notebook `02` exerce para a base original: ele inspeciona cobertura e consistência, analisa distribuições e então gera a base estendida pronta para regressão.

Os artefatos produzidos são:

- `data/TMDB_movies_processed_tmdb_extended.csv`
- figuras em `code/tmdb/images/`

### 4. Geração dos folds de revenue

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

### 5. Seleção de modelos e busca de hiperparâmetros

O notebook [code/revenue/02_model_grid_search.ipynb](code/revenue/02_model_grid_search.ipynb) executa a comparação inicial entre os regressores nas duas versões da variável-alvo: `Sem transformação` e `Com log1p`.

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
- consolidação de `MSE`, `RMSE`, `MAE` e `R^2` na escala original em dólares, inclusive para a variante com `log1p`, após reconversão das predições.

Os artefatos produzidos são:

- `data/revenue_model_selection/model_selection_results.csv`
- `data/revenue_model_selection/model_selection_predictions.csv`
- `data/revenue_model_selection/model_selection_summary.csv`

### 6. Análise detalhada do melhor modelo sem transformação

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

### 7. Análise detalhada do melhor modelo com log1p

O notebook [code/revenue/04_analysis_log1p.ipynb](code/revenue/04_analysis_log1p.ipynb) repete a mesma leitura analítica para a versão `Com log1p`, permitindo comparar diretamente como a transformação do alvo afeta resíduos, erro por faixa de receita, dispersão entre valores reais e previstos e importância de atributos.

Os artefatos produzidos são:

- `data/revenue_model_selection/error_analysis/04_analysis_log1p_metricas_por_faixa_raw.csv`
- `data/revenue_model_selection/error_analysis/04_analysis_log1p_metricas_por_faixa_formatado.csv`
- figuras em `code/revenue/images/`

### 8. Regressão com a base TMDB estendida sem transformação

O notebook [code/revenue/05_tmdb_extended_no_transform.ipynb](code/revenue/05_tmdb_extended_no_transform.ipynb) reaplica o mesmo protocolo de validação externa do baseline, mas usando `data/TMDB_movies_processed_tmdb_extended.csv` e mantendo apenas a versão `Sem transformação` da variável-alvo.

Essa etapa reaproveita exatamente:

- os mesmos `10` folds salvos em `data/revenue_folds.csv`;
- os mesmos limites de estratificação em `data/revenue_stratification_bins.csv`;
- a mesma grade de modelos e hiperparâmetros do notebook `02`.

Além do novo grid search, o notebook compara o desempenho com o baseline original e produz uma análise de erros específica para a base estendida.

Os artefatos produzidos são:

- `data/revenue_model_selection_tmdb_extended_no_transform/model_selection_results.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/model_selection_predictions.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/model_selection_summary.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/baseline_comparison_summary.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/error_analysis/`
- figuras em `code/revenue/images/`

### 9. Funções de perda mais robustas

O notebook [code/revenue/06_tmdb_extended_robust_losses.ipynb](code/revenue/06_tmdb_extended_robust_losses.ipynb) investiga se a principal limitação do melhor modelo global do cenário TMDB estendido pode ser atenuada apenas pela troca da função de perda.

No fluxo atual do projeto, essa etapa não reabre a comparação completa entre famílias. Em vez disso, ela seleciona automaticamente a família vencedora do notebook `05` e compara apenas variantes robustas dessa mesma família. Com os artefatos atualmente salvos no repositório, isso significa testar o `XGBoost` com objetivos `squared_error`, `absolute_error` e `pseudo-Huber`.

O objetivo é verificar se losses menos sensíveis a desvios extremos ajudam a lidar melhor com a forte assimetria da variável-alvo e com os maiores erros concentrados nas faixas superiores de arrecadação.

Os artefatos produzidos são:

- `data/revenue_robust_losses_tmdb_extended_promising/model_selection_results.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/model_selection_predictions.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/model_selection_summary.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/best_robust_vs_global_best.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/error_analysis/06_robust_losses_metricas_por_faixa.csv`

### 10. Modelos por faixa de arrecadação

O notebook [code/revenue/07_tmdb_extended_oracle_band_models.ipynb](code/revenue/07_tmdb_extended_oracle_band_models.ipynb) investiga o quanto modelos locais por faixa poderiam ajudar a reduzir o erro. Para isso, ele treina regressões especializadas por faixa de receita e usa a faixa **verdadeira** do conjunto de teste apenas como oracle de análise.

Essa etapa é diagnóstica: ela não representa um pipeline de predição utilizável em produção, mas mostra o ganho potencial de especializar o ajuste em regiões mais homogêneas da distribuição. Ela funciona como justificativa metodológica para os notebooks `08` e `09`, que tentam aproximar essa ideia com fluxos realistas de classificação + regressão.

Os artefatos produzidos são:

- `data/revenue_oracle_band_models_tmdb_extended/model_selection_results.csv`
- `data/revenue_oracle_band_models_tmdb_extended/model_selection_predictions.csv`
- `data/revenue_oracle_band_models_tmdb_extended/model_selection_summary.csv`
- `data/revenue_oracle_band_models_tmdb_extended/comparison_with_global_models.csv`
- `data/revenue_oracle_band_models_tmdb_extended/training_band_sizes.csv`
- `data/revenue_oracle_band_models_tmdb_extended/error_analysis/07_oracle_band_models_metricas_por_faixa.csv`

### 11. Abordagem híbrida classificação + regressão

O notebook [code/revenue/08_tmdb_extended_classification_plus_regression.ipynb](code/revenue/08_tmdb_extended_classification_plus_regression.ipynb) implementa um pipeline em dois estágios: primeiro um classificador prevê a faixa de arrecadação do filme, depois um regressor local estima o valor contínuo dentro da faixa prevista.

Essa estratégia tenta transformar a assimetria de `revenue` em um problema hierárquico, no qual a escolha do regime de arrecadação antecede a regressão final.

Os artefatos produzidos são:

- `data/revenue_hybrid_classification_regression_tmdb_extended/model_selection_results.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/model_selection_predictions.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/model_selection_summary.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/best_hybrid_vs_global_best.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/training_band_sizes.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/error_analysis/08_classification_plus_regression_metricas_por_faixa.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/error_analysis/08_classification_plus_regression_confusion_matrix.csv`

### 12. Abordagem híbrida com soft routing

O notebook [code/revenue/09_tmdb_extended_soft_routing.ipynb](code/revenue/09_tmdb_extended_soft_routing.ipynb) mantém a etapa de classificação em faixas, mas substitui o roteamento duro do notebook `08` por uma combinação ponderada das previsões dos regressores locais usando as probabilidades previstas pelo classificador.

Essa variação foi criada para testar se erros pequenos de classificação entre faixas vizinhas podem ser suavizados sem recorrer ao oracle do notebook `07`.

Os artefatos produzidos são:

- `data/revenue_soft_routing_tmdb_extended/model_selection_results.csv`
- `data/revenue_soft_routing_tmdb_extended/model_selection_predictions.csv`
- `data/revenue_soft_routing_tmdb_extended/model_selection_summary.csv`
- `data/revenue_soft_routing_tmdb_extended/best_soft_routing_vs_global_and_oracle.csv`
- `data/revenue_soft_routing_tmdb_extended/training_band_sizes.csv`
- `data/revenue_soft_routing_tmdb_extended/classifier_confidence_summary.csv`
- `data/revenue_soft_routing_tmdb_extended/error_analysis/09_soft_routing_metricas_por_faixa.csv`
- `data/revenue_soft_routing_tmdb_extended/error_analysis/09_soft_routing_confusion_matrix.csv`

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

### Base adicional de metadados do TMDB

O arquivo `data/TMDB_movies_additional_metadata.csv` contém `6918` filmes e `19` colunas. Ele registra, para cada `id_tmdb` já presente na base processada, informações adicionais de lançamento e produção coletadas de forma incremental.

Os campos desse estágio incluem:

- data de estreia e atributos calendáricos derivados;
- diretor principal;
- nomes do elenco principal;
- empresa principal de produção e conjunto de empresas associadas;
- contagens agregadas de elenco e de empresas.

### Base estendida para regressão

O arquivo `data/TMDB_movies_processed_tmdb_extended.csv` contém `6918` filmes e `287` colunas.

Além dos atributos já existentes em `data/TMDB_movies_processed.csv`, essa versão incorpora:

- variáveis numéricas como ano, mês, quantidade de empresas e quantidade de membros do elenco;
- indicadores binários de período de estreia;
- codificação binária dos diretores mais frequentes;
- codificação binária dos membros de elenco mais frequentes;
- codificação binária das empresas de produção mais frequentes.

Essa é a base usada nos notebooks `05` a `09`, que estudam o efeito de um cenário preliminar mais informado e estratégias adicionais para lidar com a assimetria da receita.

---

## Protocolo de validação e métricas

A validação do projeto foi desenhada para preservar comparabilidade entre modelos e evitar particionamentos arbitrários.

O fluxo atual é:

- `revenue` é discretizada em `5` faixas com `pd.qcut`;
- os bins servem apenas para estratificar os folds;
- a regressão é executada tanto sobre o valor bruto de `revenue` quanto sobre `log1p(revenue)`;
- os folds externos são fixados com `random_state=222050006`;
- os hiperparâmetros são escolhidos com `GridSearchCV` dentro de cada fold de treino;
- as métricas finais são sempre calculadas na escala original em dólares.

Os arquivos consolidados registram:

- `MSE`
- `RMSE`
- `MAE`
- `R^2`

No estado atual dos artefatos salvos em `data/revenue_model_selection/model_selection_summary.csv`, há `16` linhas consolidadas: `8` para `Sem transformação` e `8` para `Com log1p`.

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

Se quiser refazer a coleta do zero, crie um `.env` na raiz do projeto a partir do template:

```bash
cp .env.template .env
```

Depois, preencha a chave:

```python
TMDB_API_KEY="YOUR_KEY"
```

O fluxo recomendado é manter a chave em `.env`, que é o arquivo carregado automaticamente pelos utilitários do TMDB. Se esse arquivo não existir, o código ainda aceita `TMDB_API_KEY` como variável de ambiente de fallback. O arquivo `.env.template` serve apenas como molde versionado para criar o `.env`.

Se você pretende apenas reproduzir a parte de modelagem a partir dos CSVs já presentes, essa etapa é opcional.

### 3. Executar os notebooks na ordem correta

Fluxo completo:

```text
code/tmdb/01_movies.ipynb
code/tmdb/02_analysis_and_preprocessing.ipynb
code/tmdb/03_additional_metadata_enrichment.ipynb
code/tmdb/04_analysis_and_preprocessing_additional_metadata.ipynb
code/revenue/01_fold_generation.ipynb
code/revenue/02_model_grid_search.ipynb
code/revenue/03_analysis_no_transform.ipynb
code/revenue/04_analysis_log1p.ipynb
code/revenue/05_tmdb_extended_no_transform.ipynb
code/revenue/06_tmdb_extended_robust_losses.ipynb
code/revenue/07_tmdb_extended_oracle_band_models.ipynb
code/revenue/08_tmdb_extended_classification_plus_regression.ipynb
code/revenue/09_tmdb_extended_soft_routing.ipynb
```

Fluxo mínimo para reproduzir a regressão a partir dos dados já salvos:

```text
code/tmdb/02_analysis_and_preprocessing.ipynb
code/revenue/01_fold_generation.ipynb
code/revenue/02_model_grid_search.ipynb
code/revenue/03_analysis_no_transform.ipynb
code/revenue/04_analysis_log1p.ipynb
code/revenue/05_tmdb_extended_no_transform.ipynb
```

Se o objetivo for apenas inspecionar os resultados consolidados que já estão versionados no repositório, você pode abrir diretamente:

```text
code/revenue/03_analysis_no_transform.ipynb
code/revenue/04_analysis_log1p.ipynb
code/revenue/05_tmdb_extended_no_transform.ipynb
code/revenue/06_tmdb_extended_robust_losses.ipynb
code/revenue/07_tmdb_extended_oracle_band_models.ipynb
code/revenue/08_tmdb_extended_classification_plus_regression.ipynb
code/revenue/09_tmdb_extended_soft_routing.ipynb
```

### 4. Investigar estratégias para a assimetria da receita

Os experimentos adicionais de `revenue` usam a mesma base TMDB estendida do notebook `05` e não exigem novas credenciais além da chave já usada pelo TMDB.

Fluxo recomendado em notebooks:

```text
code/tmdb/03_additional_metadata_enrichment.ipynb
code/tmdb/04_analysis_and_preprocessing_additional_metadata.ipynb
code/revenue/05_tmdb_extended_no_transform.ipynb
code/revenue/06_tmdb_extended_robust_losses.ipynb
code/revenue/07_tmdb_extended_oracle_band_models.ipynb
code/revenue/08_tmdb_extended_classification_plus_regression.ipynb
code/revenue/09_tmdb_extended_soft_routing.ipynb
```

O fluxo acima cria apenas novos artefatos associados à base estendida e às análises de assimetria.

---

## Documentos versionados

Além dos artefatos gerados pelo pipeline, o repositório também mantém em `documents/` os arquivos versionados de entrega do trabalho:

- `documents/tp1/apresentacao_da_versao_parcial_do_artigo.pdf`
- `documents/tp1/versao_final_artigo.pdf`
- `documents/tp1/versao_parcial_artigo.pdf`
- `documents/tp2/versao_final_artigo_parte2.pdf`

Esses arquivos são mantidos no repositório como documentos versionados e não fazem parte da geração automática dos notebooks.

---

## Saídas geradas

Ao longo do pipeline, o projeto grava:

- `data/TMDB_movies_original.csv`
- `data/TMDB_movies_processed.csv`
- `data/TMDB_movies_additional_metadata.csv`
- `data/TMDB_movies_processed_tmdb_extended.csv`
- `data/revenue_folds.csv`
- `data/revenue_stratification_bins.csv`
- `data/revenue_model_selection/model_selection_results.csv`
- `data/revenue_model_selection/model_selection_predictions.csv`
- `data/revenue_model_selection/model_selection_summary.csv`
- `data/revenue_model_selection/error_analysis/03_analysis_no_transform_metricas_por_faixa_raw.csv`
- `data/revenue_model_selection/error_analysis/03_analysis_no_transform_metricas_por_faixa_formatado.csv`
- `data/revenue_model_selection/error_analysis/04_analysis_log1p_metricas_por_faixa_raw.csv`
- `data/revenue_model_selection/error_analysis/04_analysis_log1p_metricas_por_faixa_formatado.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/model_selection_results.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/model_selection_predictions.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/model_selection_summary.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/baseline_comparison_summary.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/error_analysis/05_tmdb_extended_no_transform_metricas_por_faixa_raw.csv`
- `data/revenue_model_selection_tmdb_extended_no_transform/error_analysis/05_tmdb_extended_no_transform_metricas_por_faixa_formatado.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/model_selection_results.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/model_selection_predictions.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/model_selection_summary.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/best_robust_vs_global_best.csv`
- `data/revenue_robust_losses_tmdb_extended_promising/error_analysis/06_robust_losses_metricas_por_faixa.csv`
- `data/revenue_oracle_band_models_tmdb_extended/model_selection_results.csv`
- `data/revenue_oracle_band_models_tmdb_extended/model_selection_predictions.csv`
- `data/revenue_oracle_band_models_tmdb_extended/model_selection_summary.csv`
- `data/revenue_oracle_band_models_tmdb_extended/comparison_with_global_models.csv`
- `data/revenue_oracle_band_models_tmdb_extended/training_band_sizes.csv`
- `data/revenue_oracle_band_models_tmdb_extended/error_analysis/07_oracle_band_models_metricas_por_faixa.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/model_selection_results.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/model_selection_predictions.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/model_selection_summary.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/best_hybrid_vs_global_best.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/training_band_sizes.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/error_analysis/08_classification_plus_regression_metricas_por_faixa.csv`
- `data/revenue_hybrid_classification_regression_tmdb_extended/error_analysis/08_classification_plus_regression_confusion_matrix.csv`
- `data/revenue_soft_routing_tmdb_extended/model_selection_results.csv`
- `data/revenue_soft_routing_tmdb_extended/model_selection_predictions.csv`
- `data/revenue_soft_routing_tmdb_extended/model_selection_summary.csv`
- `data/revenue_soft_routing_tmdb_extended/best_soft_routing_vs_global_and_oracle.csv`
- `data/revenue_soft_routing_tmdb_extended/training_band_sizes.csv`
- `data/revenue_soft_routing_tmdb_extended/classifier_confidence_summary.csv`
- `data/revenue_soft_routing_tmdb_extended/error_analysis/09_soft_routing_metricas_por_faixa.csv`
- `data/revenue_soft_routing_tmdb_extended/error_analysis/09_soft_routing_confusion_matrix.csv`
- figuras em `code/tmdb/images/`
- figuras em `code/revenue/images/`

---

## Análise em notebook

Os notebooks [code/revenue/03_analysis_no_transform.ipynb](code/revenue/03_analysis_no_transform.ipynb), [code/revenue/04_analysis_log1p.ipynb](code/revenue/04_analysis_log1p.ipynb), [code/revenue/05_tmdb_extended_no_transform.ipynb](code/revenue/05_tmdb_extended_no_transform.ipynb), [code/revenue/06_tmdb_extended_robust_losses.ipynb](code/revenue/06_tmdb_extended_robust_losses.ipynb), [code/revenue/07_tmdb_extended_oracle_band_models.ipynb](code/revenue/07_tmdb_extended_oracle_band_models.ipynb), [code/revenue/08_tmdb_extended_classification_plus_regression.ipynb](code/revenue/08_tmdb_extended_classification_plus_regression.ipynb) e [code/revenue/09_tmdb_extended_soft_routing.ipynb](code/revenue/09_tmdb_extended_soft_routing.ipynb) concentram a leitura final dos resultados. Eles permitem explorar:

- o ranking médio dos modelos avaliados;
- a distribuição dos resíduos do melhor modelo;
- o comportamento do erro por faixa de receita;
- a dispersão entre valores reais e preditos;
- a importância média das features por permutação.
