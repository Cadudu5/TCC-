"""Gera um manifesto das marcações de neutrófilos recuperadas sem retreinar."""

from __future__ import annotations

import argparse
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image

from features.extract import extraction_metadata, segment_superpixels


PROJECT_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(value: str) -> str:
    value = value.removeprefix("rotulos_")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def best_image(labels_path: Path, images: list[Path]) -> Path:
    target = canonical_name(labels_path.stem)
    ranked = sorted(
        ((SequenceMatcher(None, target, canonical_name(path.stem)).ratio(), path) for path in images),
        key=lambda item: (-item[0], str(item[1])),
    )
    score, path = ranked[0]
    if score < 0.70:
        raise FileNotFoundError(f"Imagem correspondente incerta para {labels_path.name} (score={score:.2f})")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-slic", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "neutrofilos" / "manifest_recuperado.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_paths = sorted((PROJECT_ROOT / "marcacoes-neutrofilos").glob("*.csv"))
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    all_images = [
        path
        for directory in (PROJECT_ROOT / "imagens", PROJECT_ROOT / "Neutrofilos")
        for path in directory.iterdir()
        if path.suffix.lower() in image_extensions
    ]
    unique_images: dict[str, Path] = {}
    for path in all_images:
        unique_images.setdefault(sha256(path), path)
    images = list(unique_images.values())

    files: list[dict[str, object]] = []
    total_rows = total_positive = 0
    for labels_path in labels_paths:
        labels = pd.read_csv(labels_path).sort_values("superpixel_id")
        image_path = best_image(labels_path, images)
        rows = int(len(labels))
        positives = int((labels["label"] == 1).sum())
        total_rows += rows
        total_positive += positives
        slic_validated = False
        if args.validate_slic:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            generated_ids = np.unique(segment_superpixels(image))
            expected_ids = labels["superpixel_id"].to_numpy(dtype=int)
            if not np.array_equal(generated_ids, expected_ids):
                raise ValueError(f"IDs SLIC incompatíveis: {labels_path.name} x {image_path.name}")
            slic_validated = True
        files.append(
            {
                "labels": str(labels_path.relative_to(PROJECT_ROOT)),
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "labels_sha256": sha256(labels_path),
                "image_sha256": sha256(image_path),
                "rows": rows,
                "positive": positives,
                "negative": rows - positives,
                "slic_ids_validated": slic_validated,
            }
        )

    model_path = PROJECT_ROOT / "models" / "artifacts" / "rf_best_cv.joblib"
    metadata_path = PROJECT_ROOT / "models" / "artifacts" / "rf_best_cv_meta.json"
    manifest = {
        "status": "partial_ground_truth_use_existing_model_do_not_retrain",
        "recovered_files": len(files),
        "recovered_rows": total_rows,
        "recovered_positive": total_positive,
        "recovered_negative": total_rows - total_positive,
        "article_rows_after_background_removal": 94_999,
        "article_positive": 12_862,
        "article_negative": 82_137,
        "missing_rows_relative_to_article": 94_999 - total_rows,
        "missing_positive_relative_to_article": 12_862 - total_positive,
        "extraction": extraction_metadata(),
        "preserved_model": {
            "path": str(model_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(model_path),
            "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
            "metadata_sha256": sha256(metadata_path),
        },
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Marcações recuperadas: {len(files)}")
    print(f"Superpixels: {total_rows} (negativos={total_rows-total_positive}, positivos={total_positive})")
    print(f"Manifesto: {args.output}")


if __name__ == "__main__":
    main()
