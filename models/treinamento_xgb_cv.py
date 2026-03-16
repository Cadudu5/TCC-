import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise RuntimeError("xgboost não está instalado. Execute: pip install xgboost") from exc


# --- CONFIGURAÇÕES ---
DATASET_PATH = '../dataset_balanceado_sem_fundo.csv'
N_SPLITS = 5
RANDOM_STATE = 42

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}


def carregar_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de dataset não encontrado em '{path}'")
    print(f"Carregando dataset de '{path}'...")
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("O dataset precisa conter a coluna 'label'.")
    if 'superpixel_id' not in df.columns:
        raise ValueError("O dataset precisa conter a coluna 'superpixel_id'.")
    return df
    if __name__ == '__main__':
        main()
def definir_grupos_features(df: pd.DataFrame) -> dict:
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

    for nome, cols in experimentos.items():
        if not cols:
            print(f"Aviso: experimento '{nome}' não possui colunas. Será ignorado.")
    return {k: v for k, v in experimentos.items() if v}


def _safe_drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing)


def avaliar_fold(y_true, y_pred) -> dict:
    return {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=['Fundo (0)', 'Neutrófilo (1)']),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


def treinar_cross_validation(experimentos: dict, X: pd.DataFrame, y: pd.Series):
    resultados_summary = {}
    tabela_metricas = []
    melhor_f1 = -1.0
    melhor_modelo = None
    melhor_features = []
    melhor_nome_exp = ""
    melhor_confusao_media = None
    fold_train_sizes = []
    fold_test_sizes = []

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    primeira_chave = next(iter(experimentos))

    for nome_exp, features in experimentos.items():
        print("\n" + "=" * 50)
        print(f"EXECUTANDO EXPERIMENTO (CV): {nome_exp}")
        print("=" * 50)

        X_subset = X[features].to_numpy()
        y_array = y.to_numpy()

        fold_metrics = []
        fold_idx = 1
        confusion_matrices = []

        for train_idx, test_idx in cv.split(X_subset, y_array):
            print(f"\n--- Fold {fold_idx}/{N_SPLITS} ---")

            X_train_fold, X_test_fold = X_subset[train_idx], X_subset[test_idx]
            y_train_fold, y_test_fold = y_array[train_idx], y_array[test_idx]

            if nome_exp == primeira_chave:
                fold_train_sizes.append(len(train_idx))
                fold_test_sizes.append(len(test_idx))

            num_pos = int((y_train_fold == 1).sum())
            num_neg = int((y_train_fold == 0).sum())
            scale_pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0

            params = dict(XGB_PARAMS)
            params['scale_pos_weight'] = scale_pos_weight

            modelo = XGBClassifier(**params)
            modelo.fit(X_train_fold, y_train_fold)

            y_pred_fold = modelo.predict(X_test_fold)
            metrics = avaliar_fold(y_test_fold, y_pred_fold)

            print(metrics['classification_report'])
            print("Matriz de Confusão:")
            print(metrics['confusion_matrix'])

            fold_metrics.append({
                'fold': fold_idx,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy'],
            })
            confusion_matrices.append(metrics['confusion_matrix'])

            fold_idx += 1

        precisions = [m['precision'] for m in fold_metrics]
        recalls = [m['recall'] for m in fold_metrics]
        f1s = [m['f1'] for m in fold_metrics]
        accs = [m['accuracy'] for m in fold_metrics]

        resumo = {
            'precision_mean': float(np.mean(precisions)),
            'precision_std': float(np.std(precisions)),
            'recall_mean': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
        }

        resultados_summary[nome_exp] = resumo

        for m in fold_metrics:
            tabela_metricas.append({
                'modelo': nome_exp,
                'fold': m['fold'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'accuracy': m['accuracy'],
            })

        if resumo['f1_mean'] > melhor_f1:
            melhor_f1 = resumo['f1_mean']
            num_pos_total = int((y_array == 1).sum())
            num_neg_total = int((y_array == 0).sum())
            scale_pos_weight_global = float(num_neg_total / num_pos_total) if num_pos_total > 0 else 1.0
            params = dict(XGB_PARAMS)
            params['scale_pos_weight'] = scale_pos_weight_global

            melhor_modelo = XGBClassifier(**params)
            melhor_modelo.fit(X_subset, y_array)
            melhor_features = list(features)
            melhor_nome_exp = nome_exp
            melhor_confusao_media = np.mean(confusion_matrices, axis=0)

    return (
        resultados_summary,
        tabela_metricas,
        melhor_modelo,
        melhor_features,
        melhor_nome_exp,
        melhor_f1,
        fold_train_sizes,
        fold_test_sizes,
        melhor_confusao_media,
    )


def imprimir_resumos(resultados_summary: dict):
    print("\n" + "#" * 60)
    print("        RESUMO FINAL (Cross-Validation)")
    print("#" * 60)
    header = (
        f"{'Modelo':<40} | "
        f"{'Precision (m±dp)':<20} | "
        f"{'Recall (m±dp)':<18} | "
        f"{'F1 (m±dp)':<15} | "
        f"{'Accuracy (m±dp)':<20}"
    )
    print(header)
    print("-" * len(header))

    ordenado = sorted(resultados_summary.items(), key=lambda item: item[1]['f1_mean'], reverse=True)
    for nome, resumo in ordenado:
        precision_text = f"{resumo['precision_mean']:.4f} ± {resumo['precision_std']:.4f}"
        recall_text = f"{resumo['recall_mean']:.4f} ± {resumo['recall_std']:.4f}"
        f1_text = f"{resumo['f1_mean']:.4f} ± {resumo['f1_std']:.4f}"
        acc_text = f"{resumo['accuracy_mean']:.4f} ± {resumo['accuracy_std']:.4f}"
        print(
            f"{nome:<40} | "
            f"{precision_text:<20} | "
            f"{recall_text:<18} | "
            f"{f1_text:<15} | "
            f"{acc_text:<20}"
        )


def salvar_artifacts(modelo, features, melhor_nome_exp, melhor_f1, tabela_metricas):
    print("\nSalvando melhor modelo e metadados...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, 'xgb_best_cv.joblib')
    meta_path = os.path.join(artifacts_dir, 'xgb_best_cv_meta.json')
    metrics_path = os.path.join(artifacts_dir, 'xgb_cv_metricas_por_fold.csv')
    fi_path = os.path.join(artifacts_dir, 'xgb_cv_feature_importances.csv')

    joblib.dump(modelo, model_path)

    slic_params = {
        "n_segments": 5000,
        "compactness": 10,
        "sigma": 3,
    }

    meta = {
        "best_experiment": melhor_nome_exp,
        "feature_names": features,
        "xgb_params": dict(XGB_PARAMS),
        "cv": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "slic": slic_params,
        "f1_mean_cv": melhor_f1,
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    try:
        pd.DataFrame(tabela_metricas).to_csv(metrics_path, index=False)
        print(f"Tabela de métricas por fold salva em: {metrics_path}")
    except Exception as exc:
        print(f"Aviso: não foi possível salvar a tabela de métricas: {exc}")

    booster = modelo.get_booster()
    gain_scores = booster.get_score(importance_type='gain')
    feature_to_gain = {name: float(gain_scores.get(name, 0.0)) for name in features}
    df_importances = (
        pd.DataFrame({'feature': list(feature_to_gain.keys()), 'gain': list(feature_to_gain.values())})
        .sort_values('gain', ascending=False)
        .reset_index(drop=True)
    )
    df_importances.to_csv(fi_path, index=False)

    print(f"Melhor experimento (média F1): {melhor_nome_exp} | F1={melhor_f1:.4f}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Metadados salvos em: {meta_path}")
    print(f"Importâncias salvas em: {fi_path}")

    top_k = min(20, len(df_importances))
    print("\nTop atributos por ganho (top 20):")
    for i in range(top_k):
        row = df_importances.iloc[i]
        print(f"{i + 1:>2}. {row['feature']}: {row['gain']:.6f}")


def main():
    df = carregar_dataset(DATASET_PATH)
    experimentos = definir_grupos_features(df)
    if not experimentos:
        print("Nenhum experimento válido encontrado (listas de features vazias). Abortando.")
        return

    X = _safe_drop_columns(df, ['label', 'superpixel_id', 'image_origin'])
    y = df['label']

    (
        resultados_summary,
        tabela_metricas,
        melhor_modelo,
        melhor_features,
        melhor_nome_exp,
        melhor_f1,
        fold_train_sizes,
        fold_test_sizes,
        melhor_confusao_media,
    ) = treinar_cross_validation(
        experimentos,
        X,
        y,
    )

    total_amostras = len(y)
    if fold_train_sizes and fold_test_sizes:
        mean_train = int(np.round(np.mean(fold_train_sizes)))
        mean_test = int(np.round(np.mean(fold_test_sizes)))
        print("\n" + "-" * 60)
        print(f"Total de amostras utilizadas: {total_amostras}")
        print(f"Amostras por fold - treino: {fold_train_sizes} (média ≈ {mean_train})")
        print(f"Amostras por fold - teste: {fold_test_sizes} (média ≈ {mean_test})")
        print("-" * 60)

    imprimir_resumos(resultados_summary)

    if melhor_modelo is not None:
        salvar_artifacts(melhor_modelo, melhor_features, melhor_nome_exp, melhor_f1, tabela_metricas)
        if melhor_confusao_media is not None:
            print("\nMatriz de Confusão Média (melhor experimento):")
            print(np.array2string(
                melhor_confusao_media,
                formatter={'float_kind': lambda value: f"{value:8.2f}"}
            ))


if __name__ == '__main__':
    main()
