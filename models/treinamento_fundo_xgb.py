import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'fundo_enriquecido')
DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DEFAULT_MODEL_NAME = 'fundo_xgb.joblib'
DEFAULT_META_NAME = 'fundo_xgb_meta.json'


def load_fundo_dataset(
    data_dir: str,
    color_means_only: bool = False,
    include_glcm: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")

    csvs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {data_dir}")

    frames = []
    for path in sorted(csvs):
        df = pd.read_csv(path)
        if 'label' not in df.columns:
            raise ValueError(f"Arquivo {path} não contém coluna 'label'")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    full = full[~full['label'].isna()].copy()
    full['label'] = full['label'].astype(int)

    drop_cols = []
    for c in ['superpixel_id']:
        if c in full.columns:
            drop_cols.append(c)
    feature_cols = [c for c in full.columns if c not in drop_cols + ['label']]

    if color_means_only:
        desired = [
            'rgb_mean_ch1', 'rgb_mean_ch2', 'rgb_mean_ch3',
            'hsv_mean_ch1', 'hsv_mean_ch2', 'hsv_mean_ch3',
            'lab_mean_ch1', 'lab_mean_ch2', 'lab_mean_ch3',
        ]
        if include_glcm:
            desired += ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_correlation']
        feature_cols = [c for c in desired if c in full.columns]

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(full[c])]
    data = full[num_cols + ['label']].copy()

    if data.empty or data.shape[1] <= 1:
        raise ValueError("Conjunto de dados vazio ou sem features numéricas válidas.")

    # Balanceamento das classes por downsampling aleatório da classe majoritária
    class_counts = data['label'].value_counts()
    if len(class_counts) > 1:
        min_count = class_counts.min()
        balanced_parts = []
        for label_value, count in class_counts.items():
            subset = data[data['label'] == label_value]
            if count > min_count:
                subset = subset.sample(n=min_count, random_state=42)
            balanced_parts.append(subset)
        data = pd.concat(balanced_parts, ignore_index=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        raise ValueError("O dataset de fundo precisa conter pelo menos duas classes para balanceamento.")

    return data, num_cols


def train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> XGBClassifier:
    num_pos = int((y_train == 1).sum())
    num_neg = int((y_train == 0).sum())
    scale_pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0

    clf = XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


def maybe_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    cv_folds: int,
) -> dict:
    if cv_folds and cv_folds > 1:
        clf = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
        return {
            'cv_metric': 'f1_weighted',
            'cv_folds': cv_folds,
            'cv_scores': scores.tolist(),
            'cv_mean': float(np.mean(scores)),
            'cv_std': float(np.std(scores)),
        }
    return {}


def save_artifacts(model, meta: dict, artifacts_dir: str, model_name: str, meta_name: str):
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, model_name)
    meta_path = os.path.join(artifacts_dir, meta_name)

    joblib.dump(model, model_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {model_path}")
    print(f"Metadados salvos em: {meta_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Treinamento - Classificador de Fundo (XGBoost)')
    p.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Diretório com CSVs enriquecidos de fundo')
    p.add_argument('--artifacts-dir', type=str, default=DEFAULT_ARTIFACTS_DIR, help='Diretório para salvar artefatos')
    p.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME, help='Nome do arquivo do modelo .joblib')
    p.add_argument('--meta-name', type=str, default=DEFAULT_META_NAME, help='Nome do arquivo de metadados .json')
    p.add_argument('--n-estimators', type=int, default=300, help='Número de árvores (boosting rounds) do XGBoost')
    p.add_argument('--max-depth', type=int, default=6, help='Profundidade máxima das árvores')
    p.add_argument('--learning-rate', type=float, default=0.1, help='Taxa de aprendizado')
    p.add_argument('--subsample', type=float, default=0.8, help='Subsample de instâncias por árvore')
    p.add_argument('--colsample-bytree', type=float, default=0.8, help='Subsample de features por árvore')
    p.add_argument('--test-size', type=float, default=0.2, help='Proporção para o conjunto de teste')
    p.add_argument('--cv-folds', type=int, default=0, help='Número de folds para validação cruzada (0 desabilita)')
    p.add_argument('--dry-run', action='store_true', help='Apenas carrega e imprime info do dataset, sem treinar')
    p.add_argument('--color-means-only', action='store_true', default=True, help='Usar apenas médias de cor RGB/HSV/LAB (sem desvios)')
    p.add_argument('--include-glcm', action='store_true', default=False, help='Quando combinado com --color-means-only, adiciona GLCM (contrast, dissimilarity, homogeneity, correlation)')
    return p.parse_args()


def main():
    args = parse_args()

    print('--- Treinamento do Classificador de Fundo (XGBoost) ---')
    print(f"Dados: {os.path.abspath(args.data_dir)}")

    data, feature_cols = load_fundo_dataset(
        args.data_dir,
        color_means_only=args.color_means_only,
        include_glcm=args.include_glcm,
    )
    print(f"Dataset carregado: {data.shape[0]} amostras, {len(feature_cols)} features")

    X = data.drop(columns=['label'])
    y = data['label'].astype(int)

    if args.dry_run:
        classes, counts = np.unique(y, return_counts=True)
        print('Distribuição de classes:', dict(zip(map(int, classes), map(int, counts))))
        print('Dry run concluído. Nada a treinar.')
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    xgb_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'logloss',
    }

    model = train_xgb_classifier(X_train, y_train, xgb_params)

    metrics = evaluate_model(model, X_test, y_test)

    cv_info = maybe_cross_validate(X, y, xgb_params, args.cv_folds)

    classes, counts = np.unique(y, return_counts=True)
    print('\n--- Resumo ---')
    print('Amostras totais:', int(len(y)))
    print('Distribuição de classes:', dict(zip(map(int, classes), map(int, counts))))
    print('Acurácia:', f"{metrics['accuracy']:.4f}")
    print('F1 ponderado:', f"{metrics['f1_weighted']:.4f}")
    print('F1 macro:', f"{metrics['f1_macro']:.4f}")
    print('Matriz de confusão:', metrics['confusion_matrix'])

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        'created_at': now,
        'algorithm': 'XGBClassifier',
        'params': xgb_params,
        'data_dir': os.path.abspath(args.data_dir),
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1]),
        'test_size': args.test_size,
        'metrics': metrics,
        'cv': cv_info,
        'feature_names': list(feature_cols),
    }

    save_artifacts(model, meta, args.artifacts_dir, args.model_name, args.meta_name)


if __name__ == '__main__':
    main()
