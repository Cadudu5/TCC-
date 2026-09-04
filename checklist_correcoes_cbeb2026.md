# Guia de Correções e Levantamento de Informações — CBEB 2026

Este documento sistematiza de forma estruturada todas as ações, levantamentos de dados e justificativas textuais necessárias para a revisão final do artigo submetido ao **CBEB 2026**.

---

## 1. Perguntas e Dados Concretos a Levantar
*Preencha ou resgate as respostas e parâmetros abaixo para inseri-los nas seções de Métodos e Resultados.*

### A. Quantificação da Amostra Histológica (Seção II)
1. **Quantos animais murinos** forneceram as amostras utilizadas no estudo?
2. **Quantas lâminas histológicas** físicas foram processadas?
3. **Quantos campos microscópicos (imagens)** foram capturados no total a partir dessas lâminas?
4. **Qual foi o aumento óptico** utilizado no microscópio (ex.: 20×, 40×) e qual a **resolução média** das imagens capturadas (em pixels)?

### B. Hiperparâmetros do Algoritmo SLIC (Seção II-A)
5. **Qual foi o número aproximado de segmentos** configurado no algoritmo SLIC (`n_segments` ou tamanho médio de superpixel)?
6. **Qual o valor do parâmetro de compacidade** (`compactness`) utilizado?
7. **Houve aplicação de suavização prévia** (ex.: filtro gaussiano com parâmetro $\sigma$)? Se sim, qual valor foi utilizado?

### C. Protocolo do Padrão-Ouro e Anotação Especializada (Seção II-B e II-C)
8. **Quantos especialistas** participaram da rotulagem dos superpixels (pesquisadores da FORP-USP)?
9. **Como foi conduzida a anotação manual?**
   - Foi feita por um único especialista sênior com calibração prévia?
   - Ou múltiplos especialistas anotaram em consenso / de forma independente?
10. **Houve avaliação de concordância** intraobservador ou interobservador? *(Se não houve, essa informação será enquadrada na Discussão como trabalho futuro).*

### D. Validação Cruzada e Distribuição dos Superpixels (Seção II-D)
11. **Como os superpixels foram distribuídos entre os 5 folds?**
   - A estratificação foi realizada no nível individual de superpixels misturados?
   - Ou foi agrupada por imagem/lâmina? *(Se foi no nível de superpixels, basta confirmar para que o texto explique com transparência e aponte a validação agrupada como trabalho futuro).*

### E. Desempenho Computacional da Ferramenta Gráfica (Seção III-D)
12. **Qual é o tempo médio de processamento** por imagem completa na interface gráfica (segmentação SLIC + extração de atributos + classificação XGBoost e Random Forest)?
13. **Em qual configuração básica de hardware** esse tempo foi obtido (ex.: processador Intel Core i5/i7, 8GB/16GB RAM, execução em CPU convencional)?

---

## 2. Observações, Explicações e Justificativas Textuais a Inserir
*Estas são as argumentações, moderações conceituais e análises qualitativas que devem ser incorporadas ao texto do manuscrito.*

### A. Moderação do Tom da Escrita (Introdução, Discussão, Conclusão)
* **Observação do Revisor:** O texto trazia afirmações excessivamente categóricas sem ter realizado medições diretas de tempo humano ou concordância entre múltiplos observadores.
* **Ajuste Textual:**
  * Substituir termos como *"garantir consistência estatística"* e *"melhorar a precisão e eficiência"* por formulações moderadas, tais como: *"potencial para apoiar a padronização"*, *"mitigar a variabilidade subjetiva visual"* e *"fornecer ferramenta de suporte à triagem"*.
  * Deixar explícito que os ganhos de agilidade e reprodutibilidade são hipóteses e benefícios esperados da automação, e não conclusões decorrentes de estudo comparativo cronometrado.

### B. Posicionamento da Novidade e Diferencial do Estudo (Introdução e Discussão)
* **Observação do Revisor:** Algoritmos clássicos de ML e SLIC são conhecidos na literatura; o diferencial precisa ficar mais nítido.
* **Ajuste Textual:**
  * Explicitar que a contribuição científica central do trabalho não reside na proposição de uma nova arquitetura matemática de aprendizado de máquina, mas sim:
    1. Na **modelagem em duas etapas sequenciais** (remoção de fundo $\rightarrow$ quantificação de neutrófilos em tecido);
    2. Na aplicação inédita ao **tecido periodontal apical murino com imuno-histoquímica (DAB)**;
    3. Na integração de ponta a ponta em uma **ferramenta gráfica aberta e acessível** voltada a pesquisadores da área.

### C. Caracterização da Tabela I como Estudo de Ablação (Seção III-A)
* **Observação do Revisor:** Avaliar a contribuição individual de cada conjunto de atributos.
* **Ajuste Textual:**
  * Destacar no texto que a **Tabela I já constitui um estudo de ablação experimental**, demonstrando o ganho incremental obtido ao unir informações cromáticas e estruturais:
    - Textura isolada: Acurácia de 0.733 e AUC de 0.795.
    - CIELAB isolado: Acurácia de 0.888 e AUC de 0.956.
    - **CIELAB + Textura (proposta final):** Acurácia de 0.919 e AUC de 0.972.

### D. Justificativa Objetiva para Seleção do Random Forest (Seção III-C e Discussão)
* **Observação do Revisor:** Outros modelos (XGBoost, MLP, SVM) apresentaram AUC ou sensibilidade ligeiramente superiores; por que escolher o Random Forest?
* **Ajuste Textual:**
  * Explicitar o critério de priorização clínica pré-definido:
    - O Random Forest alcançou a **maior especificidade (0.909)** e a **maior acurácia global (0.919)** na Etapa 2.
    - No contexto histopatológico, priorizar a especificidade reduz a ocorrência de falsos positivos (234 amostras), evitando a superestimação da área inflamada, o que é fundamental para a correta avaliação da resposta tecidual.

### E. Análise Qualitativa de Erros da Matriz de Confusão (Seção III-C e Discussão)
* **Observação do Revisor:** Discutir as causas dos 234 falsos positivos e 184 falsos negativos da Tabela V.
* **Ajuste Textual:**
  * **Falsos Positivos (234 superpixels):** Atribuídos a artefatos pontuais de precipitação do cromógeno DAB, regiões com acúmulo de hemossiderina/hemácias extravasadas ou feixes densos de fibras colágenas que apresentam intensidade cromática semelhante à reação positiva.
  * **Falsos Negativos (184 superpixels):** Atribuídos a neutrófilos com imunorreatividade mais tênue/fraca ou células localizadas nas extremidades dos superpixels, cuja média de cor do segmento foi atenuada pelos pixels adjacentes de estroma.

### F. Justificativa do Uso de Aprendizado de Máquina Clássico vs. Deep Learning (Discussão)
* **Observação do Revisor:** Posicionar as vantagens da abordagem frente a redes neurais convolucionais (CNNs).
* **Ajuste Textual:**
  * Explicar que a combinação de SLIC + classificadores tabulares clássicos foi uma decisão deliberada de engenharia:
    - Adequada a bases de imagens experimentais moderadas, mitigando o risco de sobreajuste (*overfitting*) inerente a redes profundas sem grandes volumes de dados anotados.
    - Baixo custo computacional e dispensabilidade de placas gráficas dedicadas (GPU), viabilizando o uso direto em computadores convencionais de laboratório.
    - Maior interpretabilidade direta dos atributos físicos extraídos (canais de cor e métricas GLCM).

### G. Discussão sobre Subamostragem (*Undersampling*) (Discussão)
* **Observação do Revisor:** A redução drástica da classe majoritária pode impactar a representatividade real.
* **Ajuste Textual:**
  * Justificar que a subamostragem foi adotada na fase de treino para equalizar a fronteira de decisão e evitar o viés da classe dominante.
  * Assumir como oportunidade para estudos futuros a comparação com técnicas de ponderação de classes (*class weighting*), sobreamostragem (SMOTE) ou abordagens híbridas.

### H. Limitações Inerentes aos Superpixels e Generalização da Base (Discussão)
* **Observação do Revisor:** Apontar limites de generalização e desafios do SLIC.
* **Ajuste Textual:**
  * **Limitações do SLIC:** Regiões com aglomeração celular densa (*cell clumping*), sobreposição de núcleos ou limites difusos entre estroma e fundo podem gerar superpixels híbridos (contendo simultaneamente tecido sadio e infiltrado).
  * **Generalização:** Reconhecer explicitamente que o método foi avaliado em uma base preliminar originada de um único protocolo experimental institucional, apontando a validação externa (outras colorações, microscópios e centros de pesquisa) e a validação agrupada por animal (*GroupKFold*) como direcionamentos futuros.

---

## 3. Resumo Visual do Roteiro de Edição por Seção

| Seção do Artigo | Itens a Preencher / Ajustar |
| :--- | :--- |
| **Resumo** | Suavizar termos afirmativos de eficiência e precisão; contextualizar como ferramenta de apoio. |
| **I. Introdução** | Moderar linguagem; destacar diferencial (fluxo em 2 etapas + ferramenta para periodontite apical). |
| **II. Métodos** | Inserir $N$ de animais, lâminas, fotos, resolução; parâmetros do SLIC; protocolo de rotulagem e CV. |
| **III. Resultados** | Nomear Tabela I como ablação; justificar escolha do RF; detalhar causas de FP/FN; inserir tempo em segundos. |
| **IV. Discussão** | Discutir trade-off clássico vs. Deep Learning; limitações do SLIC; undersampling; generalização externa e GroupKFold. |
| **V. Conclusão** | Alinhar conclusões finais com a moderação do tom e perspectivas de validação expandida. |
