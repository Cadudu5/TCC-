# Reprodução da Etapa 1 do artigo

Data da reprodução: 15 de agosto de 2026.

## Dados

- 19 imagens recuperadas e validadas.
- 92.340 superpixels antes do balanceamento.
- 89.640 não fundo e 2.700 fundo.
- 5.400 instâncias após subamostragem, 2.700 por classe.
- Atributos: médias dos três canais CIELAB e quatro descritores GLCM.
- Validação: StratifiedKFold, cinco folds, shuffle e random state 42.

## Resultados

| Modelo | Acurácia reproduzida | Artigo | Sensibilidade reproduzida | Artigo | Especificidade reproduzida | Artigo | AUC reproduzida | Artigo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,9278 | 0,928 | 0,9326 | 0,934 | 0,9230 | 0,923 | 0,9803 | 0,981 |
| Random Forest | 0,9267 | 0,927 | 0,9281 | 0,926 | 0,9252 | 0,928 | 0,9799 | 0,981 |
| MLP | 0,9256 | 0,924 | 0,9415 | 0,935 | 0,9096 | 0,913 | 0,9795 | 0,978 |
| SVM | 0,9172 | 0,917 | 0,9322 | 0,931 | 0,9022 | 0,904 | 0,9727 | 0,973 |

O XGBoost apresentou a maior acurácia, reproduzindo a escolha da Etapa 1 do
artigo. A maior diferença absoluta foi 0,0065, na sensibilidade da MLP.

## Rastreabilidade

- Dataset completo SHA-256:
  `ec2661f625cf31855bd7d54467cdff93730524e45a6a0c01f6a349133419109f`
- Dataset balanceado SHA-256:
  `06af6f60de326fb1812c5cd346c952d6a78459a4d40ade69cac173f80375ad1e`
- Manifesto SHA-256 no momento da geração:
  `94a2d82f002606a4ba094f84a95f8652824799b7e1ff90f239bd88d07fea9b12`

Os hashes individuais das imagens e marcações constam em
`data/processed/fundo/manifest.json`. Os valores não devem ser transferidos
manualmente para o artigo sem conferir o arquivo JSON gerado e o estado do Git.

## Limite atual

Esta reprodução cobre a classificação de fundo. A Etapa 2 continua usando o RF
recuperado para inferência, mas suas métricas não são reproduzíveis até que as
sete marcações faltantes sejam recuperadas ou refeitas.
