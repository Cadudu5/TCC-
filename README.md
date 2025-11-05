## Objetivo do projeto
Treinar um modelo de aprendizado de máquina para identificar superpixels positivos para neutrófilos em amostras histológicas de tecido gengival murino.

## Ambiente e dependências
- **Python**: 3.9+ (recomendado)
- **Instalação de dependências**:
```bash
pip install -r requirements.txt
```

## Rotulador de superpixels (`rotulador_gui.py`)
Ferramenta GUI para segmentar a imagem em superpixels (SLIC), extrair características e rotular manualmente superpixels positivos para neutrófilos.

### Parâmetros SLIC usados
- **N_SEGMENTS**: 5000
- **COMPACTNESS**: 10
- **SIGMA**: 3

Mantenha esses valores consistentes em todo o pipeline (especialmente ao recalcular superpixels fora da GUI).

### O que é extraído por superpixel
- **Cores (médias)**: `rgb_mean_ch{1..3}`, `hsv_mean_ch{1..3}`, `lab_mean_ch{1..3}`
- **Textura (GLCM)**: `glcm_contrast`, `glcm_dissimilarity`, `glcm_homogeneity`, `glcm_correlation`
- **Metadados**: `superpixel_id` e coluna de rótulo `label` (0=fundo, 1=neutrófilo)

### Como usar
```bash
python rotulador_gui.py
```
1. Clique em "Carregar Imagem" e selecione um arquivo (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`).
2. Aguarde a segmentação e a extração de características.
3. Clique nos superpixels para alternar o rótulo: verde = positivo (neutrófilo), transparente = negativo.
4. Clique em "Salvar Resultados" para exportar:
   - CSV com as características + `label` (nome sugerido: `dataset_<nome_da_imagem>.csv`)
   - Imagem com marcações (PNG/JPG)

## Preparação do dataset para treino
Você pode construir um dataset único a partir de vários CSVs gerados pela GUI.

### Opção A) Unir múltiplos CSVs rotulados
1. Coloque os arquivos CSV individuais na pasta `datasets_individuais/`.
2. Execute:
```bash
python unir_datasets.py
```
Isso criará `dataset_completo.csv` na raiz do projeto e adicionará a coluna `image_origin` com a origem de cada amostra.

### Opção B) Recalcular características e juntar com rótulos existentes
Se você possui um CSV de rótulos separado por `superpixel_id`, use `enriquecer_csv.py`:
- Ajuste `IMAGE_PATH`, `LABELS_CSV_PATH` e `OUTPUT_CSV_PATH` no script.
- Garanta que `N_SEGMENTS`, `COMPACTNESS` e `SIGMA` sejam exatamente os mesmos da GUI.
```bash
python enriquecer_csv.py
```

### Colunas esperadas pelo treino
- `label` (0/1), `superpixel_id`, `image_origin` (se unificado por `unir_datasets.py`)
- Features de cor: prefixos `rgb_`, `hsv_`, `lab_`
- Features de textura: prefixo `glcm_`
  
Observação: se seu dataset final não tiver a coluna `image_origin`, adicione-a (por exemplo, preenchendo com o nome da imagem de origem) ou edite `models/treinamento.py` para não tentar remover essa coluna em `df.drop(...)`.

### Utilitário para balancear classes (`utils/balancear_classes.py`)
Use este script quando precisar reduzir o desbalanceamento entre classes 0/1 antes do treino.

```bash
# Exemplo: balancear dataset_completo.csv usando a coluna 'label'
python utils/balancear_classes.py --input dataset_completo.csv --output dataset_balanceado.csv --label-col label --random-state 42
```

O script realiza uma subamostragem aleatória da classe majoritária para igualar o número de amostras da classe minoritária e gera um novo CSV balanceado.

## Treinamento e avaliação (`models/treinamento.py`)
Script que executa experimentos comparando grupos de features e reporta métricas.

### Entrada
- Por padrão lê `../dataset_completo.csv` (relativo à pasta `models/`).
  - Se necessário, ajuste `DATASET_PATH` no início do script.

### Como executar
```bash
cd models
python treinamento.py
```

### O que o script faz
- Divide os dados em treino/teste (25%, estratificado; `random_state=42`).
- Treina `RandomForestClassifier` com `class_weight='balanced'`.
- Executa 6 experimentos:
  1) Apenas RGB
  2) Apenas HSV
  3) Apenas LAB
  4) Apenas Textura (GLCM)
  5) Todas as Cores (RGB+HSV+LAB)
  6) Completo (Cor + Textura)
- Imprime `classification_report`, matriz de confusão e um resumo final com F1-Score ponderado por experimento.

### Ajustes úteis
- Número de árvores: `N_ESTIMATORS`
- Balanceamento: `CLASS_WEIGHT`
- Reprodutibilidade: `RANDOM_STATE`

## Treinamento do modelo de fundo (`models/treinamento_fundo_rf.py`)
Classificador binário que identifica superpixels de fundo (1=fundo, 0=não-fundo). A GUI usa esse modelo no botão "Marcar fundo" para pintar o fundo em vermelho e, depois, ignorá-lo na classificação de neutrófilos.

### Dados de entrada
- Por padrão, lê os CSVs de `fundo_enriquecido/` (cada arquivo com colunas por superpixel e `label` 0/1).
- Os parâmetros SLIC devem ser consistentes com os usados na extração das features.

### Grupos de features suportados
- Médias de cor (RGB/HSV/LAB)
- Médias de cor + textura (GLCM: contrast, dissimilarity, homogeneity, correlation)

### Como executar
```bash
# Apenas médias RGB/HSV/LAB (padrão)
python models/treinamento_fundo_rf.py --color-means-only --n-estimators 300 --cv-folds 5

# Médias + GLCM
python models/treinamento_fundo_rf.py --color-means-only --include-glcm --n-estimators 300 --cv-folds 5

# (Opcional) especificar diretório de dados
python models/treinamento_fundo_rf.py --data-dir caminho/para/csvs --color-means-only --include-glcm
```

### Saída
- Artefatos salvos em `models/artifacts/`:
  - `fundo_rf.joblib`: modelo RandomForest ajustado
  - `fundo_rf_meta.json`: metadados, incluindo a lista `feature_names` usada na inferência

### Uso na GUI
- Ao iniciar `apps/inferencia_gui.py`, clique em "Marcar fundo". A GUI:
  - Extrai as mesmas features por superpixel
  - Carrega `models/artifacts/fundo_rf.joblib` e usa somente as colunas listadas em `fundo_rf_meta.json`
  - Pinta o fundo em vermelho e, em seguida, "Analisar imagem" ignora esses superpixels

## Executáveis (opcional)
Na pasta `dist/` existem versões `.exe` do rotulador para Windows (`Rotulador.exe` e `Rotulador_lite.exe`). Podem ser usados para rotular sem precisar iniciar o Python, mas o fluxo de treino permanece o mesmo (gerar CSVs e unificá-los).

## Observações importantes
- Garanta consistência dos parâmetros SLIC entre a rotulagem e qualquer recalculo posterior de superpixels.
- Dados de histologia tendem a ser desbalanceados; o uso de `class_weight='balanced'` ajuda a mitigar isso.
- Se necessário, normalize a coloração previamente (ver `normalizar_imagem.py`).

## Referência rápida
```bash
# 1) Rotular e exportar CSVs individuais
python rotulador_gui.py

# 2) Unificar CSVs em um único dataset
python unir_datasets.py  # gera dataset_completo.csv

# 3) Treinar e avaliar modelos
cd models
python treinamento.py

# 4) Treinar e salvar melhor modelo (neutrófilos)
cd models
python treinamento_rf.py  # salva models/artifacts/rf_best.joblib e rf_best_meta.json

# 5) Treinar o modelo de fundo (exemplos)
cd ..
python models/treinamento_fundo_rf.py --color-means-only --cv-folds 5
python models/treinamento_fundo_rf.py --color-means-only --include-glcm --cv-folds 5

# 6) Rodar GUI de inferência e salvar overlay
cd ..
python apps/inferencia_gui.py
```