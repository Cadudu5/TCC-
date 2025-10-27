# Documentação de Bibliotecas do Projeto

Este documento descreve todas as bibliotecas Python utilizadas no projeto de detecção de neutrófilos, suas versões e objetivos.

## Bibliotecas Principais

### Processamento de Imagens

#### **scikit-image** (v0.25.2)
- **Objetivo**: Biblioteca principal para processamento de imagens científicas
- **Uso no projeto**:
  - Segmentação de superpixels usando algoritmo SLIC
  - Conversão entre espaços de cor (RGB, HSV, LAB, Grayscale)
  - Extração de características de textura (GLCM - Gray Level Co-occurrence Matrix)
  - Marcação de bordas de superpixels
  - Manipulação e transformação de imagens
- **Módulos usados**: `segmentation`, `color`, `feature`, `io`, `util`

#### **opencv-python** (v4.12.0.88)
- **Objetivo**: Biblioteca de visão computacional
- **Uso no projeto**:
  - Leitura e escrita de imagens
  - Normalização de cor em imagens histológicas (via staintools)

#### **staintools** (v2.1.2)
- **Objetivo**: Normalização de coloração em imagens histopatológicas
- **Uso no projeto**:
  - Padronização de cores entre diferentes lâminas histológicas
  - Correção de variações na coloração H&E
- **Arquivo**: `normalizar_imagem.py`

#### **Pillow (PIL)** (v11.3.0)
- **Objetivo**: Manipulação de imagens
- **Uso no projeto**:
  - Interface gráfica: carregamento e exibição de imagens
  - Conversão de formatos de imagem
- **Arquivos**: `rotulador_lite.py`, `inferencia_gui.py`

#### **imageio** (v2.37.0)
- **Objetivo**: Leitura e escrita de imagens em diversos formatos
- **Uso no projeto**: Suporte adicional para formatos de imagem (TIF, PNG, JPG)

#### **tifffile** (v2025.6.11)
- **Objetivo**: Leitura específica de arquivos TIFF microscópicos
- **Uso no projeto**: Carregamento de imagens histológicas em formato .tif

---

### Machine Learning e Análise de Dados

#### **scikit-learn** (v1.7.1)
- **Objetivo**: Framework principal para machine learning
- **Uso no projeto**:
  - **Modelos**: Random Forest, SVM, MLP (Redes Neurais)
  - **Preprocessamento**: StandardScaler para normalização de features
  - **Avaliação**: Métricas (accuracy, precision, recall, F1-score, confusion matrix)
  - **Validação**: train_test_split para divisão treino/teste
- **Arquivos**: `treinamento_rf.py`, `treinamento_svm.py`, `treinamento_nn.py`

#### **xgboost** (v3.0.5)
- **Objetivo**: Algoritmo de Gradient Boosting otimizado
- **Uso no projeto**:
  - Treinamento de modelo XGBClassifier para classificação de neutrófilos
  - Alternativa aos modelos Random Forest e SVM
- **Arquivo**: `treinamento_xgb.py`

#### **joblib** (v1.5.1)
- **Objetivo**: Serialização eficiente de objetos Python
- **Uso no projeto**:
  - Salvamento e carregamento de modelos treinados (.pkl)
  - Persistência de pipelines de preprocessamento
- **Arquivos**: Todos os scripts de treinamento e inferência

---

### Manipulação de Dados

#### **pandas** (v2.3.1)
- **Objetivo**: Análise e manipulação de dados tabulares
- **Uso no projeto**:
  - Armazenamento de features extraídas dos superpixels
  - Gerenciamento de rótulos (labels) positivos/negativos
  - Carregamento e salvamento de datasets em CSV
  - União e balanceamento de datasets
- **Arquivos**: Todos os scripts do projeto

#### **numpy** (v2.2.6)
- **Objetivo**: Computação numérica e arrays multidimensionais
- **Uso no projeto**:
  - Manipulação de imagens como arrays
  - Operações matemáticas nas features
  - Processamento de máscaras de superpixels
  - Salvamento/carregamento de mapas de superpixels (.npy)
- **Arquivos**: Todos os scripts do projeto

---

### Visualização

#### **matplotlib** (v3.10.3)
- **Objetivo**: Visualização de dados e imagens
- **Uso no projeto**:
  - Interface de rotulação manual de superpixels
  - Exibição de imagens segmentadas
  - Visualização de resultados e métricas
  - Integração com Tkinter (FigureCanvasTkAgg)
- **Arquivos**: `rotulador.py`, `rotulador_lite.py`, `rotulador_gui.py`

---

### Interface Gráfica

#### **tkinter** (built-in)
- **Objetivo**: Biblioteca padrão Python para GUI
- **Uso no projeto**:
  - Interface gráfica para rotulação de superpixels
  - Aplicação de inferência com visualização de overlays
  - Seleção de arquivos e exibição de resultados
- **Arquivos**: `rotulador_lite.py`, `rotulador_gui.py`, `inferencia_gui.py`

---

### Utilitários

#### **tqdm** (v4.67.1)
- **Objetivo**: Barras de progresso para loops
- **Uso no projeto**:
  - Feedback visual durante extração de features
  - Monitoramento de processamento de múltiplas imagens
- **Arquivos**: `rotulador.py`, `enriquecer_csv.py`

#### **scipy** (v1.16.1)
- **Objetivo**: Algoritmos científicos e matemáticos
- **Uso no projeto**:
  - Dependência do scikit-image e scikit-learn
  - Operações matemáticas avançadas

---

## Bibliotecas de Suporte

### Desenvolvimento e Empacotamento

- **pyinstaller** (v6.15.0): Criação de executáveis standalone
- **pyinstaller-hooks-contrib** (v2025.8): Hooks adicionais para PyInstaller

### Jupyter/IPython (Desenvolvimento)

- **ipykernel** (v6.30.0): Kernel Jupyter para notebooks
- **ipython** (v9.4.0): Shell interativo Python
- **jupyter_client** (v8.6.3): Cliente Jupyter
- **jupyter_core** (v5.8.1): Funcionalidades core do Jupyter

### Outras Dependências

- **python-dateutil** (v2.9.0.post0): Manipulação de datas
- **pytz** (v2025.2): Fusos horários
- **networkx** (v3.5): Dependência do scikit-image
- **threadpoolctl** (v3.6.0): Controle de threads em bibliotecas numéricas

---

## Resumo por Funcionalidade

| Funcionalidade | Bibliotecas Principais |
|----------------|------------------------|
| **Segmentação de Imagens** | scikit-image, numpy |
| **Extração de Features** | scikit-image, numpy, pandas |
| **Normalização de Cor** | staintools, opencv-python |
| **Rotulação Manual** | matplotlib, tkinter, pandas |
| **Treinamento de Modelos** | scikit-learn, xgboost, joblib |
| **Inferência** | scikit-learn, joblib, tkinter |
| **Manipulação de Dados** | pandas, numpy |
| **Visualização** | matplotlib, PIL |
| **Empacotamento** | pyinstaller |

---

## Versões Críticas

As seguintes versões são **críticas** para reprodutibilidade dos resultados:

- **scikit-image**: 0.25.2
- **scikit-learn**: 1.7.1
- **numpy**: 2.2.6
- **pandas**: 2.3.1
- **xgboost**: 3.0.5

## Instalação

Para instalar todas as dependências necessárias:

```bash
pip install -r requirements.txt
```

**Nota**: O arquivo `requirements.txt` atual contém apenas as bibliotecas principais. Para um ambiente completo, considere gerar um novo requirements com:

```bash
pip freeze > requirements_completo.txt
```

---

*Última atualização: 24 de outubro de 2025*
