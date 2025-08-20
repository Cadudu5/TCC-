import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- CONFIGURAÇÕES ---
DATASET_PATH = '../dataset_completo.csv'

# Parâmetros do Modelo Random Forest
N_ESTIMATORS = 100
RANDOM_STATE = 42 # Para reprodutibilidade
CLASS_WEIGHT = 'balanced' # Essencial para dados desbalanceados

def run_training_experiments():
    """
    Função principal que carrega os dados e executa todos os experimentos
    de treinamento e avaliação.
    """
    # --- 1. Carregamento e Preparação dos Dados ---
    if not os.path.exists(DATASET_PATH):
        print(f"ERRO: Arquivo de dataset não encontrado em '{DATASET_PATH}'")
        return

    print(f"Carregando dataset de '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)

    # --- 2. Definição dos Grupos de Features para os Experimentos ---
    features_rgb = [col for col in df.columns if col.startswith('rgb')]
    features_hsv = [col for col in df.columns if col.startswith('hsv')]
    features_lab = [col for col in df.columns if col.startswith('lab_')] 
    print("Colunas selecionadas para LAB:", features_lab)
    features_cor = features_rgb + features_hsv + features_lab
    features_textura = [col for col in df.columns if col.startswith('glcm')]
    features_completo = features_cor + features_textura

    experimentos = {
        "1. Apenas RGB": features_rgb,
        "2. Apenas HSV": features_hsv,
        "3. Apenas LAB": features_lab,
        "4. Apenas Textura": features_textura,
        "5. Todas as Cores": features_cor,
        "6. Modelo Completo (Cor + Textura)": features_completo
    }

    # Separando as features (X) do alvo (y)
    X = df.drop(columns=['label', 'superpixel_id', 'image_origin'])
    y = df['label']

    # --- 3. Divisão em Treino e Teste (FEITA UMA ÚNICA VEZ) ---
    # Isso garante que todos os modelos sejam avaliados no mesmo conjunto de teste.
    # `stratify=y` mantém a proporção de classes nos conjuntos de treino e teste.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nDados divididos em {len(X_train)} amostras de treino e {len(X_test)} de teste.")

    # Dicionário para armazenar os resultados para comparação final
    resultados_finais = {}

    # --- 4. Loop Experimental ---
    for nome_exp, features_exp in experimentos.items():
        print("\n" + "="*50)
        print(f"EXECUTANDO EXPERIMENTO: {nome_exp}")
        print("="*50)

        # Seleciona as colunas de features para este experimento
        X_train_subset = X_train[features_exp]
        X_test_subset = X_test[features_exp]

        # Inicializa o modelo com os parâmetros definidos
        modelo = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight=CLASS_WEIGHT,
            n_jobs=-1 
        )

        print(f"Treinando o modelo com {len(features_exp)} features...")
        modelo.fit(X_train_subset, y_train)

        print("Avaliando o modelo no conjunto de teste...")
        y_pred = modelo.predict(X_test_subset)

        # Imprime o relatório de classificação detalhado
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Fundo (0)', 'Neutrófilo (1)']))
        
        print("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))

        # Armazena o F1-Score ponderado para o resumo final
        f1_ponderado = f1_score(y_test, y_pred, average='weighted')
        resultados_finais[nome_exp] = f1_ponderado

    # --- 5. Análise Final ---
    print("\n" + "#"*60)
    print("        RESUMO FINAL DOS EXPERIMENTOS (F1-Score)")
    print("#"*60)
    
    # Ordena os resultados do melhor para o pior
    resultados_ordenados = sorted(resultados_finais.items(), key=lambda item: item[1], reverse=True)
    
    print(f"{'Modelo':<40} | {'F1-Score (Ponderado)':<20}")
    print("-"*60)
    for nome, score in resultados_ordenados:
        print(f"{nome:<40} | {score:<20.4f}")
    print("-"*60)


if __name__ == '__main__':
    run_training_experiments()