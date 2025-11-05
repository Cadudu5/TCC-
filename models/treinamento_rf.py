import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# --- CONFIGURAÇÕES ---
# dataset_completo 
# dataset_completo_sem_sd
DATASET_PATH = '../dataset_balanceado.csv'

# Parâmetros do Modelo Random Forest
N_ESTIMATORS = 100
RANDOM_STATE = 42 # Para reprodutibilidade
CLASS_WEIGHT = 'balanced' # Essencial para dados balanceados

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
    X = df.drop(columns=['label', 'superpixel_id'])
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
    # Lista para montar a tabela de métricas por experimento
    tabela_metricas = []

    # Controle do melhor modelo ao longo dos experimentos
    melhor_f1 = -1.0
    melhor_modelo = None
    melhor_features = []
    melhor_nome_exp = ""

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

        # Calcula métricas agregadas
        precision_ponderado = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_ponderado = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_ponderado = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        acuracia = accuracy_score(y_test, y_pred)

        # Armazena métricas na tabela
        tabela_metricas.append({
            'modelo': nome_exp,
            'precision': precision_ponderado,
            'recall': recall_ponderado,
            'f1': f1_ponderado,
            'accuracy': acuracia,
        })

        # Armazena o F1-Score ponderado para o resumo final
        resultados_finais[nome_exp] = f1_ponderado

        # Atualiza o melhor modelo se este experimento superar o F1 atual
        if f1_ponderado > melhor_f1:
            melhor_f1 = f1_ponderado
            melhor_modelo = modelo
            melhor_features = list(features_exp)
            melhor_nome_exp = nome_exp

    # --- 5. Tabela de Métricas por Experimento ---
    print("\n" + "#"*60)
    print("        TABELA DE MÉTRICAS POR EXPERIMENTO")
    print("#"*60)
    print(f"{'Modelo':<40} | {'Precision':>9} | {'Recall':>7} | {'F1':>5} | {'Acurácia':>9}")
    print("-"*86)
    for r in tabela_metricas:
        print(f"{r['modelo']:<40} | {r['precision']:>9.4f} | {r['recall']:>7.4f} | {r['f1']:>5.4f} | {r['accuracy']:>9.4f}")

    # --- 6. Análise Final ---
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

    # --- 7. Salvamento do melhor modelo e metadados ---
    if melhor_modelo is not None:
        print("\nSalvando o melhor modelo e metadados...")
        # Garante que salvaremos sempre em uma pasta 'artifacts' ao lado deste script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(base_dir, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(artifacts_dir, 'rf_best.joblib')
        meta_path = os.path.join(artifacts_dir, 'rf_best_meta.json')

        joblib.dump(melhor_modelo, model_path)

        # Parâmetros SLIC usados no pipeline de rotulagem (README)
        slic_params = {
            "n_segments": 5000,
            "compactness": 10,
            "sigma": 3,
        }

        meta = {
            "best_experiment": melhor_nome_exp,
            "feature_names": melhor_features,
            "model_params": {
                "n_estimators": N_ESTIMATORS,
                "class_weight": CLASS_WEIGHT,
                "random_state": RANDOM_STATE,
            },
            "slic": slic_params,
        }

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Melhor experimento: {melhor_nome_exp} | F1={melhor_f1:.4f}")
        print(f"Modelo salvo em: {model_path}")
        print(f"Metadados salvos em: {meta_path}")


if __name__ == '__main__':
    run_training_experiments()