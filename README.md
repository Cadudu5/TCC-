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

# 4) Treinar e salvar melhor modelo (este repo)
cd models
python treinamento_rf.py  # salva models/artifacts/rf_best.joblib e rf_best_meta.json

# 5) Rodar GUI de inferência e salvar overlay
cd ..
python apps/inferencia_gui.py
```