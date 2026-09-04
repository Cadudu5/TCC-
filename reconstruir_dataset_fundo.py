"""Reconstrói o dataset de fundo a partir das imagens e marcações recuperadas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image

from features.extract import (
    SLIC_COMPACTNESS,
    SLIC_N_SEGMENTS,
    SLIC_SIGMA,
    extract_features,
    extraction_metadata,
    segment_superpixels,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "imagens"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "marcacoes-fundo"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "fundo"
EXPECTED_ROWS = 92_340
EXPECTED_BACKGROUND = 2_700
RANDOM_STATE = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(value: str) -> str:
    value = value.removeprefix("rotulos_fundo_").removeprefix("rotulos_")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", "", value.casefold())
    # Arquivos recuperados no Windows contêm "neutr¢filos".
    return value.replace("neutrfilos", "neutrofilos")


def discover_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in image_extensions)
    labels = sorted(labels_dir.glob("*.csv"))
    image_by_name = {canonical_name(path.stem): path for path in images}

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for labels_path in labels:
        image_path = image_by_name.get(canonical_name(labels_path.stem))
        if image_path is None:
            missing.append(labels_path.name)
        else:
            pairs.append((image_path, labels_path))
    if missing:
        raise FileNotFoundError("Imagens não encontradas para: " + ", ".join(missing))
    if len(pairs) != len(images):
        paired_images = {image.resolve() for image, _ in pairs}
        extras = [image.name for image in images if image.resolve() not in paired_images]
        raise ValueError("Imagens sem marcação de fundo: " + ", ".join(extras))
    return pairs


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"superpixel_id", "label"}
    if set(labels.columns) != required:
        raise ValueError(f"{path.name}: colunas esperadas {sorted(required)}, obtidas {list(labels.columns)}")
    if labels["superpixel_id"].duplicated().any():
        raise ValueError(f"{path.name}: superpixel_id duplicado")
    labels["superpixel_id"] = labels["superpixel_id"].astype(int)
    labels["label"] = labels["label"].astype(int)
    if not set(labels["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: rótulos diferentes de 0/1")
    return labels.sort_values("superpixel_id").reset_index(drop=True)


def balance_classes(dataset: pd.DataFrame) -> pd.DataFrame:
    counts = dataset["label"].value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("O dataset precisa conter as classes 0 e 1")
    minimum = int(counts.min())
    parts = [
        dataset[dataset["label"] == label].sample(n=minimum, random_state=RANDOM_STATE)
        for label in (0, 1)
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def write_csv_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Valida pares e totais sem extrair atributos")
    parser.add_argument("--allow-unexpected-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args.images_dir.resolve(), args.labels_dir.resolve())
    labels_by_path = {path: load_labels(path) for _, path in pairs}
    raw_rows = sum(len(labels) for labels in labels_by_path.values())
    raw_background = sum(int((labels["label"] == 1).sum()) for labels in labels_by_path.values())
    raw_tissue = raw_rows - raw_background

    print(f"Pares imagem/marcação: {len(pairs)}")
    print(f"Superpixels: {raw_rows} (não fundo={raw_tissue}, fundo={raw_background})")
    if not args.allow_unexpected_counts and (raw_rows, raw_background) != (EXPECTED_ROWS, EXPECTED_BACKGROUND):
        raise ValueError(
            f"Totais diferentes do artigo: esperado ({EXPECTED_ROWS}, {EXPECTED_BACKGROUND}), "
            f"obtido ({raw_rows}, {raw_background})"
        )
    if args.dry_run:
        print("Validação concluída; nenhum dataset foi escrito.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir = args.output_dir / "por_imagem"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifest_files: list[dict[str, object]] = []

    for index, (image_path, labels_path) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {image_path.name}")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        superpixels = segment_superpixels(
            image,
            n_segments=SLIC_N_SEGMENTS,
            compactness=SLIC_COMPACTNESS,
            sigma=SLIC_SIGMA,
        )
        labels = labels_by_path[labels_path]
        generated_ids = np.unique(superpixels)
        expected_ids = labels["superpixel_id"].to_numpy()
        if not np.array_equal(generated_ids, expected_ids):
            raise ValueError(
                f"{image_path.name}: IDs SLIC não correspondem ao CSV "
                f"(gerados={len(generated_ids)}, csv={len(expected_ids)})"
            )

        features = extract_features(image, superpixels, max_workers=args.workers)
        numeric_features = features.drop(columns=["superpixel_id"]).to_numpy(dtype=float)
        if not np.isfinite(numeric_features).all():
            raise ValueError(f"{image_path.name}: extração produziu NaN ou infinito")
        enriched = features.merge(labels, on="superpixel_id", validate="one_to_one")
        enriched.insert(1, "image_origin", image_path.name)
        per_image_path = per_image_dir / f"{labels_path.stem}_enriquecido.csv"
        write_csv_atomic(enriched, per_image_path)
        frames.append(enriched)
        manifest_files.append(
            {
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "labels": str(labels_path.relative_to(PROJECT_ROOT)),
                "image_sha256": sha256(image_path),
                "labels_sha256": sha256(labels_path),
                "rows": int(len(enriched)),
                "background": int((enriched["label"] == 1).sum()),
            }
        )

    complete = pd.concat(frames, ignore_index=True)
    balanced = balance_classes(complete)
    complete_path = args.output_dir / "dataset_fundo_completo.csv"
    balanced_path = args.output_dir / "dataset_fundo_balanceado.csv"
    write_csv_atomic(complete, complete_path)
    write_csv_atomic(balanced, balanced_path)

    manifest = {
        "method": "reconstruction_from_recovered_ground_truth",
        "random_state": RANDOM_STATE,
        "extraction": extraction_metadata(),
        "complete": {
            "path": str(complete_path.relative_to(PROJECT_ROOT)),
            "rows": int(len(complete)),
            "class_counts": {str(k): int(v) for k, v in complete["label"].value_counts().sort_index().items()},
        },
        "balanced": {
            "path": str(balanced_path.relative_to(PROJECT_ROOT)),
            "rows": int(len(balanced)),
            "class_counts": {str(k): int(v) for k, v in balanced["label"].value_counts().sort_index().items()},
        },
        "files": manifest_files,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset completo: {complete_path}")
    print(f"Dataset balanceado: {balanced_path}")
    print(f"Manifesto: {manifest_path}")


if __name__ == "__main__":
    main()
