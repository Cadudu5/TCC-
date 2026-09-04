# Quantificação automatizada de infiltrado neutrofílico

Pipeline de segmentação SLIC e aprendizado de máquina para identificar fundo e
neutrófilos em imagens histológicas do tecido periodontal.

## Estado recuperado

- Etapa 1 (fundo): 19 imagens e 19 marcações completas, totalizando 92.340
  superpixels (89.640 não fundo e 2.700 fundo).
- Etapa 2 (neutrófilos): 12 marcações recuperadas. O modelo Random Forest já
  treinado foi preservado e é usado na inferência; ele **não é retreinado** com
  a base parcial.
- Os parâmetros compartilhados ficam em `features/extract.py`. Treino e
  inferência usam SLIC 5000/10/3 e GLCM com 256 níveis, distâncias 1/3/5 e
  quatro ângulos.

## Preparação no macOS

O projeto usa Python 3.12. No Mac, o XGBoost também precisa do OpenMP:

```bash
brew install libomp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

As versões críticas estão fixadas no `requirements.txt`. A normalização por
`staintools` é opcional e possui dependências separadas em
`requirements-normalizacao.txt`.

## Rotular neutrófilos

O rotulador oficial é `rotulador_lite.py`. Ele calcula somente o SLIC ao abrir
a imagem; a extração de CIELAB/GLCM acontece posteriormente no pipeline. Isso
evita a demora de dezenas de segundos e os avisos de conversão de cor do
rotulador histórico, sem alterar os IDs dos superpixels.

No macOS, dê dois cliques em `iniciar_rotulador_macos.command` ou execute:

```bash
./iniciar_rotulador_macos.command
```

No Windows, dê dois cliques em `iniciar_rotulador_windows.bat`. Os dois
inicializadores criam o ambiente Python na primeira execução, se necessário, e
usam as dependências mínimas de `requirements-rotulador.txt`.

Fluxo de marcação:

1. Clique em **Carregar imagem** e aguarde a mensagem **Pronto**.
2. Use o botão esquerdo para marcar/desmarcar um superpixel em verde; scroll
   aplica zoom, botão direito move e botão do meio restaura a vista.
3. Para continuar um trabalho anterior, carregue primeiro a mesma imagem e
   depois clique em **Abrir marcação**. O programa recusa CSVs cujos IDs não
   correspondam à imagem, evitando associações incorretas.
4. Clique em **Salvar marcação**. São gerados o CSV `superpixel_id,label` e uma
   visualização PNG na mesma pasta. `Ctrl/Cmd+S` salva e `Ctrl/Cmd+Z` desfaz.

Mantenha no nome do CSV o nome original da imagem. Para checar uma imagem sem
abrir a interface:

```bash
.venv/bin/python rotulador_lite.py --check "imagens/21 dias 20xb.tif"
```

## Reproduzir a Etapa 1 — fundo

Primeiro valide os arquivos recuperados:

```bash
.venv/bin/python reconstruir_dataset_fundo.py --dry-run
```

Reconstrua as características e os datasets completo/balanceado:

```bash
.venv/bin/python reconstruir_dataset_fundo.py --workers 8
```

Saídas:

- `data/processed/fundo/dataset_fundo_completo.csv` — 92.340 linhas;
- `data/processed/fundo/dataset_fundo_balanceado.csv` — 5.400 linhas;
- `data/processed/fundo/por_imagem/` — 19 CSVs enriquecidos;
- `data/processed/fundo/manifest.json` — parâmetros, hashes e contagens.

Execute a validação cruzada estratificada com cinco folds usando CIELAB +
textura:

```bash
.venv/bin/python models/treinamento_fundo_artigo.py
```

O script avalia XGBoost, Random Forest, MLP e SVM com acurácia, sensibilidade,
especificidade e AUC. Os modelos são salvos sem sobrescrever os artefatos
históricos:

- `models/artifacts/fundo_xgb_artigo.joblib`;
- `models/artifacts/fundo_rf_artigo.joblib`;
- `models/artifacts/fundo_mlp_artigo.joblib`;
- `models/artifacts/fundo_svm_artigo.joblib`.

Métricas completas e comparação com o artigo ficam em
`results/fundo_cv_resumo.json` e `results/fundo_cv_metricas_por_fold.csv`.

## Rodar a aplicação

```bash
.venv/bin/python apps/inferencia_gui.py
```

Fluxo obrigatório:

1. Carregar a imagem.
2. Clicar em **Marcar fundo**. A aplicação prefere o XGBoost reconstruído com
   CIELAB + textura.
3. Clicar em **Analisar imagem**. Esta etapa usa o
   `models/artifacts/rf_best_cv.joblib` existente para neutrófilos.
4. Conferir o overlay e a aba de estatísticas.
5. Salvar o overlay e o CSV, se necessário.

O percentual informado é calculado por pixels positivos sobre pixels de tecido,
excluindo o fundo. As contagens de superpixels também são exibidas, mas não são
usadas como substituto da área.

## Situação da Etapa 2 — neutrófilos

O RF existente possui 22 entradas: médias e desvios RGB/HSV/CIELAB e quatro
descritores GLCM. Ele carrega com `scikit-learn==1.7.1` e pode ser usado para
inferência.

Para revalidar as 12 marcações, suas imagens e os IDs SLIC sem treinar nada:

```bash
.venv/bin/python auditar_recuperacao_neutrofilos.py --validate-slic
```

O comando gera `data/processed/neutrofilos/manifest_recuperado.json`, incluindo
hashes do RF preservado e de cada par imagem/marcação.

O artigo descreve um modelo final CIELAB + textura com sete entradas. Esse
artefato específico não foi recuperado. Enquanto faltarem as marcações não
recuperadas:

- não retreine e não substitua `rf_best_cv.joblib`;
- não use as 12 marcações para alegar reprodução das métricas da Etapa 2;
- mantenha os novos resultados de inferência identificados como produzidos pelo
  RF recuperado de 22 atributos.

Quando as marcações restantes forem encontradas, o dataset da Etapa 2 deverá ser
reconstruído com o mesmo extrator compartilhado, removendo os superpixels de
fundo antes do balanceamento e da validação cruzada.

## Testes

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Os testes verificam paralelismo determinístico, cálculo de área por pixels e os
contratos de features dos modelos de fundo e neutrófilos.

## Windows

O código continua compatível com Windows, mas os executáveis antigos não foram
recuperados. A versão Python deve ser validada antes de gerar novos pacotes com
PyInstaller; não use um novo executável para resultados científicos sem executar
os mesmos testes e conferir os hashes dos modelos.

Para gerar o rotulador portátil em uma máquina Windows com Python 3.12, execute
`gerar_executavel_windows.bat`. O script instala o PyInstaller em um ambiente de
compilação isolado, aplica `rotulador_windows.spec` e produz:

- `dist/Rotulador_Neutrofilos/Rotulador_Neutrofilos.exe`;
- `dist/Rotulador_Neutrofilos_Windows_Executavel.zip`.

O ZIP final já recebe as nove imagens, a pasta de resultados e as instruções. A
máquina da especialista não precisa de Python; somente a máquina que gera o
executável precisa dele. O PyInstaller não gera binários Windows no macOS, por
isso esse build deve ser executado e testado em Windows.
