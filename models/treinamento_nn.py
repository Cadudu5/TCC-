import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# --- CONFIGURAÇÕES ---
# dataset_completo 
# dataset_completo_sem_sd
DATASET_PATH = '../dataset_balanceado.csv'

# Parâmetros do MLP (rede neural densa)
MLP_PARAMS = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'alpha': 1e-4,  # L2
    'learning_rate_init': 1e-3,
    'max_iter': 300,
    'early_stopping': True,
    'n_iter_no_change': 15,
    'random_state': 42,
}


def _safe_drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing)


def run_training_experiments():
    """
    Executa experimentos com MLP (StandardScaler + MLPClassifier) comparando grupos de features.
    Salva o melhor modelo e metadados.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"ERRO: Arquivo de dataset não encontrado em '{DATASET_PATH}'")
        return

    print(f"Carregando dataset de '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)

    # --- Definição de grupos de features ---
    features_rgb = [col for col in df.columns if col.startswith('rgb')]
    features_hsv = [col for col in df.columns if col.startswith('hsv')]
    features_lab = [col for col in df.columns if col.startswith('lab_')]
    features_cor = features_rgb + features_hsv + features_lab
    features_textura = [col for col in df.columns if col.startswith('glcm')]
    features_completo = features_cor + features_textura

    experimentos = {
        "1. Apenas RGB": features_rgb,
        "2. Apenas HSV": features_hsv,
        "3. Apenas LAB": features_lab,
        "4. Apenas Textura": features_textura,
        "5. Todas as Cores": features_cor,
        "6. Modelo Completo (Cor + Textura)": features_completo,
    }

    # --- Separação X/y ---
    if 'label' not in df.columns:
        print("ERRO: coluna 'label' não encontrada no dataset")
        return

    X = _safe_drop_columns(df, ['label', 'superpixel_id', 'image_origin'])
    y = df['label']

    # --- Split treino/teste único ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nDados divididos em {len(X_train)} treino e {len(X_test)} teste.")

    resultados_finais: dict[str, float] = {}
    tabela_metricas: list[dict] = []

    melhor_f1 = -1.0
    melhor_modelo: Pipeline | None = None
    melhor_features: list[str] = []
    melhor_nome_exp = ""

    # --- Loop experimental ---
    for nome_exp, features_exp in experimentos.items():
        print("\n" + "=" * 50)
        print(f"EXECUTANDO EXPERIMENTO: {nome_exp}")
        print("=" * 50)

        X_train_subset = X_train[features_exp]
        X_test_subset = X_test[features_exp]

        # Pipeline: padronização + MLP
        modelo = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("mlp", MLPClassifier(**MLP_PARAMS)),
        ])

        print(f"Treinando o modelo com {len(features_exp)} features...")
        modelo.fit(X_train_subset, y_train)

        print("Avaliando o modelo no conjunto de teste...")
        y_pred = modelo.predict(X_test_subset)

        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Fundo (0)', 'Neutrófilo (1)']))

        print("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))

        precision_ponderado = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_ponderado = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_ponderado = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        acuracia = accuracy_score(y_test, y_pred)

        tabela_metricas.append({
            'modelo': nome_exp,
            'precision': precision_ponderado,
            'recall': recall_ponderado,
            'f1': f1_ponderado,
            'accuracy': acuracia,
        })
        resultados_finais[nome_exp] = f1_ponderado

        if f1_ponderado > melhor_f1:
            melhor_f1 = f1_ponderado
            melhor_modelo = modelo
            melhor_features = list(features_exp)
            melhor_nome_exp = nome_exp

    # --- Tabela consolidada ---
    print("\n" + "#" * 60)
    print("        TABELA DE MÉTRICAS POR EXPERIMENTO (MLP)")
    print("#" * 60)
    print(f"{'Modelo':<40} | {'Precision':>9} | {'Recall':>7} | {'F1':>5} | {'Acurácia':>9}")
    print("-" * 86)
    for r in tabela_metricas:
        print(f"{r['modelo']:<40} | {r['precision']:>9.4f} | {r['recall']:>7.4f} | {r['f1']:>5.4f} | {r['accuracy']:>9.4f}")

    # --- Resumo final ---
    print("\n" + "#" * 60)
    print("        RESUMO FINAL DOS EXPERIMENTOS (F1-Score)")
    print("#" * 60)
    resultados_ordenados = sorted(resultados_finais.items(), key=lambda item: item[1], reverse=True)
    print(f"{'Modelo':<40} | {'F1-Score (Ponderado)':<20}")
    print("-" * 60)
    for nome, score in resultados_ordenados:
        print(f"{nome:<40} | {score:<20.4f}")
    print("-" * 60)

    # --- Salvamento do melhor modelo e metadados ---
    if melhor_modelo is not None:
        print("\nSalvando o melhor modelo MLP e metadados...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(base_dir, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(artifacts_dir, 'nn_best.joblib')
        meta_path = os.path.join(artifacts_dir, 'nn_best_meta.json')

        joblib.dump(melhor_modelo, model_path)

        slic_params = {
            "n_segments": 5000,
            "compactness": 10,
            "sigma": 3,
        }

        meta = {
            "best_experiment": melhor_nome_exp,
            "feature_names": melhor_features,
            "mlp_params": dict(MLP_PARAMS),
            "slic": slic_params,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Melhor experimento: {melhor_nome_exp} | F1={melhor_f1:.4f}")
        print(f"Modelo salvo em: {model_path}")
        print(f"Metadados salvos em: {meta_path}")


if __name__ == '__main__':
    run_training_experiments()


