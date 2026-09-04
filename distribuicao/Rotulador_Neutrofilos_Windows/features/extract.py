"""Extração compartilhada de superpixels e atributos.

Este módulo é a fonte única usada na reconstrução dos datasets e na inferência.
Os parâmetros de textura reproduzem os scripts que geraram os dados de treino:
GLCM com 256 níveis, distâncias 1/3/5 e quatro ângulos.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from skimage import color
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import slic


SLIC_N_SEGMENTS = 5000
SLIC_COMPACTNESS = 10.0
SLIC_SIGMA = 3.0
SLIC_START_LABEL = 1

GLCM_LEVELS = 256
GLCM_DISTANCES = (1, 3, 5)
GLCM_ANGLES = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "correlation")

ARTICLE_FEATURE_NAMES = [
    "lab_mean_ch1",
    "lab_mean_ch2",
    "lab_mean_ch3",
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_correlation",
]


def _as_rgb_float(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb deve ser HxWx3 em RGB")
    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        image = image.astype(np.float32) / max_value
    else:
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.nanmax(image)) > 1.0:
            image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def segment_superpixels(
    image_rgb: np.ndarray,
    n_segments: int = SLIC_N_SEGMENTS,
    compactness: float = SLIC_COMPACTNESS,
    sigma: float = SLIC_SIGMA,
    start_label: int = SLIC_START_LABEL,
) -> np.ndarray:
    """Segmenta uma imagem RGB com a configuração SLIC do estudo."""
    image = _as_rgb_float(image_rgb)
    if not np.isfinite(image).all():
        raise ValueError("A imagem contém valores NaN ou infinitos")

    # Algumas versões do NumPy que usam Accelerate em Macs ARM emitem avisos
    # espúrios de overflow no produto matricial executado por rgb2lab dentro do
    # SLIC, mesmo para RGB uint8 válido. A entrada e a saída são validadas aqui,
    # de modo que o aviso possa ser suprimido sem esconder dados inválidos.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        labels = slic(
            image,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=int(start_label),
            channel_axis=-1,
        )
    if labels.size == 0 or not np.isfinite(labels).all():
        raise FloatingPointError("A segmentação SLIC produziu rótulos inválidos")
    return labels.astype(np.int32, copy=False)


def _compute_glcm_features(roi_gray_uint8: np.ndarray) -> dict[str, float]:
    if roi_gray_uint8.size == 0 or roi_gray_uint8.ndim != 2:
        return {f"glcm_{name}": 0.0 for name in GLCM_PROPERTIES}

    glcm = graycomatrix(
        roi_gray_uint8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return {
        f"glcm_{name}": float(np.mean(graycoprops(glcm, name)))
        for name in GLCM_PROPERTIES
    }


def _default_workers() -> int:
    raw_value = os.environ.get("TCC_FEATURE_WORKERS", "").strip()
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


def extract_features(
    image_rgb: np.ndarray,
    labels_slic: np.ndarray,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Extrai médias/desvios RGB, HSV e CIELAB e descritores GLCM.

    A textura é calculada sobre o retângulo delimitador do superpixel, como nos
    scripts históricos de enriquecimento. ``max_workers`` existe tanto para a
    reconstrução em lote quanto para a GUI; ``1`` força execução sequencial.
    """
    labels = np.asarray(labels_slic)
    if labels.ndim != 2 or labels.shape != np.asarray(image_rgb).shape[:2]:
        raise ValueError("labels_slic deve ter as mesmas dimensões espaciais da imagem")

    image = _as_rgb_float(image_rgb)
    # NumPy/Accelerate em alguns Macs ARM pode emitir avisos espúrios no
    # produto matricial, mesmo retornando valores corretos. Suprimimos o aviso
    # e validamos explicitamente a finitude logo depois.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        hsv_image = color.rgb2hsv(image)
        lab_image = color.rgb2lab(image)
        gray_image = color.rgb2gray(image)
    if not all(np.isfinite(space).all() for space in (hsv_image, lab_image, gray_image)):
        raise FloatingPointError("Conversão de cor produziu NaN ou infinito")
    gray_uint8 = (gray_image * 255).astype(np.uint8)
    spaces = (("rgb", image), ("hsv", hsv_image), ("lab", lab_image))

    superpixel_ids = np.unique(labels).astype(np.int32)
    if superpixel_ids.size == 0:
        return pd.DataFrame()

    minimum_id = int(superpixel_ids.min())
    shifted_labels = labels.astype(np.int64) - minimum_id + 1
    object_slices = find_objects(shifted_labels)

    def compute(superpixel_id: np.int32) -> dict[str, float | int]:
        object_index = int(superpixel_id) - minimum_id
        region_slice = object_slices[object_index]
        if region_slice is None:
            raise RuntimeError(f"Superpixel {int(superpixel_id)} sem região espacial")

        region_labels = labels[region_slice]
        mask = region_labels == int(superpixel_id)
        row: dict[str, float | int] = {"superpixel_id": int(superpixel_id)}
        for name, image_space in spaces:
            pixels = image_space[region_slice][mask]
            means = pixels.mean(axis=0)
            stds = pixels.std(axis=0)
            for channel_index in range(3):
                channel = channel_index + 1
                row[f"{name}_mean_ch{channel}"] = float(means[channel_index])
                row[f"{name}_std_ch{channel}"] = float(stds[channel_index])

        row.update(_compute_glcm_features(gray_uint8[region_slice]))
        return row

    workers = _default_workers() if max_workers is None else max(1, int(max_workers))
    if workers == 1:
        rows = [compute(superpixel_id) for superpixel_id in superpixel_ids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(compute, superpixel_ids))

    return pd.DataFrame(rows).sort_values("superpixel_id").reset_index(drop=True)


def extraction_metadata() -> dict[str, object]:
    """Configuração serializável para acompanhar datasets e modelos."""
    return {
        "slic": {
            "n_segments": SLIC_N_SEGMENTS,
            "compactness": SLIC_COMPACTNESS,
            "sigma": SLIC_SIGMA,
            "start_label": SLIC_START_LABEL,
        },
        "glcm": {
            "levels": GLCM_LEVELS,
            "distances": list(GLCM_DISTANCES),
            "angles_radians": list(GLCM_ANGLES),
            "properties": list(GLCM_PROPERTIES),
            "region": "bounding_box",
        },
    }
