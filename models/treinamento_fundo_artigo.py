"""Reproduz a Etapa 1 do artigo com CIELAB + textura e CV estratificada."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.extract import ARTICLE_FEATURE_NAMES, extraction_metadata


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "fundo" / "dataset_fundo_balanceado.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "fundo" / "manifest.json"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "models" / "artifacts"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
RANDOM_STATE = 42
N_SPLITS = 5

ARTICLE_REFERENCE = {
    "xgb": {"accuracy": 0.928, "sensitivity": 0.934, "specificity": 0.923, "auc": 0.981},
    "rf": {"accuracy": 0.927, "sensitivity": 0.926, "specificity": 0.928, "auc": 0.981},
    "mlp": {"accuracy": 0.924, "sensitivity": 0.935, "specificity": 0.913, "auc": 0.978},
    "svm": {"accuracy": 0.917, "sensitivity": 0.931, "specificity": 0.904, "auc": 0.973},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(name: str):
    if name == "xgb":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        n_iter_no_change=15,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if name == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=False,
                    ),
                ),
            ]
        )
    raise ValueError(f"Modelo desconhecido: {name}")


def prediction_scores(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    return np.asarray(model.predict(features), dtype=float)


def fold_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": matrix.tolist(),
    }


def cross_validate(name: str, features: pd.DataFrame, labels: pd.Series) -> tuple[list[dict[str, object]], dict[str, object]]:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds: list[dict[str, object]] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(features, labels), start=1):
        model = build_model(name)
        train_features = features.iloc[train_indices]
        test_features = features.iloc[test_indices]
        train_labels = labels.iloc[train_indices]
        test_labels = labels.iloc[test_indices]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            model.fit(train_features, train_labels)
            predictions = np.asarray(model.predict(test_features), dtype=int)
            scores = prediction_scores(model, test_features)
        metrics = fold_metrics(test_labels, predictions, scores)
        metrics.update({"model": name, "fold": fold, "train_size": len(train_indices), "test_size": len(test_indices)})
        folds.append(metrics)

    aggregate: dict[str, object] = {}
    for metric in ("accuracy", "sensitivity", "specificity", "auc"):
        values = np.asarray([float(fold[metric]) for fold in folds])
        aggregate[metric] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std())
    aggregate["mean_confusion_matrix"] = np.mean(
        np.asarray([fold["confusion_matrix"] for fold in folds], dtype=float), axis=0
    ).tolist()
    aggregate["article_reference"] = ARTICLE_REFERENCE[name]
    aggregate["delta_from_article"] = {
        metric: float(aggregate[metric]) - ARTICLE_REFERENCE[name][metric]
        for metric in ("accuracy", "sensitivity", "specificity", "auc")
    }
    return folds, aggregate


def package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-image", "scikit-learn", "xgboost", "joblib"]
    return {package: version(package) for package in packages}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.dataset)
    missing = [name for name in ARTICLE_FEATURE_NAMES + ["label"] if name not in dataset.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")
    labels = dataset["label"].astype(int)
    class_counts = labels.value_counts().sort_index().to_dict()
    if class_counts != {0: 2700, 1: 2700}:
        raise ValueError(f"Distribuição diferente do artigo: {class_counts}")
    features = dataset[ARTICLE_FEATURE_NAMES]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("O dataset contém NaN ou infinito nas features do artigo")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    versions = package_versions()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else None
    all_fold_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "article_stage_1_background_classification",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "manifest": manifest,
        "n_samples": int(len(dataset)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "feature_names": ARTICLE_FEATURE_NAMES,
        "cv": {"n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
        "versions": versions,
        "models": {},
    }

    for name in ("xgb", "rf", "mlp", "svm"):
        print(f"Validação cruzada: {name.upper()}")
        folds, aggregate = cross_validate(name, features, labels)
        all_fold_rows.extend(
            {
                **{key: value for key, value in fold.items() if key != "confusion_matrix"},
                "confusion_matrix": json.dumps(fold["confusion_matrix"]),
            }
            for fold in folds
        )
        final_model = build_model(name)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            final_model.fit(features, labels)
        model_path = args.artifacts_dir / f"fundo_{name}_artigo.joblib"
        meta_path = args.artifacts_dir / f"fundo_{name}_artigo_meta.json"
        joblib.dump(final_model, model_path)
        metadata = {
            "created_at": summary["created_at"],
            "algorithm": type(final_model).__name__,
            "label_semantics": {"0": "nao_fundo", "1": "fundo"},
            "feature_names": ARTICLE_FEATURE_NAMES,
            "n_samples": int(len(dataset)),
            "class_counts": summary["class_counts"],
            "extraction": extraction_metadata(),
            "cv": aggregate,
            "versions": versions,
            "dataset_sha256": summary["dataset_sha256"],
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["models"][name] = aggregate
        print(
            f"  accuracy={aggregate['accuracy']:.4f} sensitivity={aggregate['sensitivity']:.4f} "
            f"specificity={aggregate['specificity']:.4f} auc={aggregate['auc']:.4f}"
        )

    fold_path = args.results_dir / "fundo_cv_metricas_por_fold.csv"
    summary_path = args.results_dir / "fundo_cv_resumo.json"
    pd.DataFrame(all_fold_rows).to_csv(fold_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Métricas por fold: {fold_path}")
    print(f"Resumo: {summary_path}")


if __name__ == "__main__":
    main()
