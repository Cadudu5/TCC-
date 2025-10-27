import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage import color
from skimage.feature import graycomatrix, graycoprops


def segment_superpixels(image_rgb: np.ndarray, n_segments: int, compactness: float, sigma: float) -> np.ndarray:
    """
    Segmenta a imagem em superpixels usando SLIC.

    Retorna um array 2D de labels (int) com o mesmo HxW da imagem.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb deve ser HxWx3 em RGB")

    # skimage.slic espera floats [0,1]
    image_float = image_rgb.astype(np.float32) / 255.0 if image_rgb.dtype != np.float32 else image_rgb

    labels = slic(
        image_float,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1,
    )
    return labels.astype(np.int32)


def _quantize_gray(gray_float: np.ndarray, levels: int = 32) -> np.ndarray:
    """
    Converte uma imagem em escala de cinza float [0,1] para níveis inteiros [0, levels-1].
    """
    gray_clipped = np.clip(gray_float, 0.0, 1.0)
    quantized = (gray_clipped * (levels - 1) + 0.5).astype(np.uint8)
    return quantized


def _compute_glcm_features(sub_gray_q: np.ndarray) -> dict:
    """
    Calcula métricas GLCM padrão em um recorte 2D quantizado (uint8) da região.
    Usa distâncias=1 e 4 ângulos; retorna média sobre ângulos.
    """
    # Evita regiões vazias
    if sub_gray_q.size == 0 or sub_gray_q.ndim != 2:
        return {
            'glcm_contrast': 0.0,
            'glcm_dissimilarity': 0.0,
            'glcm_homogeneity': 0.0,
            'glcm_correlation': 0.0,
        }

    # Distâncias e ângulos padrão
    distances = [1]
    angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]

    # Níveis
    levels = int(sub_gray_q.max()) + 1
    levels = max(levels, 2)

    glcm = graycomatrix(
        sub_gray_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    def _prop_mean(name: str) -> float:
        vals = graycoprops(glcm, name)  # shape: (len(distances), len(angles))
        return float(np.mean(vals))

    return {
        'glcm_contrast': _prop_mean('contrast'),
        'glcm_dissimilarity': _prop_mean('dissimilarity'),
        'glcm_homogeneity': _prop_mean('homogeneity'),
        'glcm_correlation': _prop_mean('correlation'),
    }


def extract_features(image_rgb: np.ndarray, labels_slic: np.ndarray) -> pd.DataFrame:
    """
    Extrai features por superpixel:
    - Médias RGB/HSV/LAB (3 canais cada)
    - GLCM (contrast, dissimilarity, homogeneity, correlation) da região em escala de cinza

    Retorna DataFrame com uma linha por superpixel e colunas:
    'superpixel_id',
    rgb_mean_ch1..3, rgb_std_ch1..3,
    hsv_mean_ch1..3, hsv_std_ch1..3,
    lab_mean_ch1..3, lab_std_ch1..3,
    glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_correlation
    """
    if labels_slic.shape[:2] != image_rgb.shape[:2]:
        raise ValueError("labels_slic deve ter as mesmas dimensões espaciais da imagem")

    img_rgb_f = image_rgb.astype(np.float32) / 255.0
    img_hsv = color.rgb2hsv(img_rgb_f)
    img_lab = color.rgb2lab(img_rgb_f)
    img_gray = color.rgb2gray(img_rgb_f)  # float [0,1]
    img_gray_q = _quantize_gray(img_gray, levels=32)

    h, w = labels_slic.shape
    unique_sp = np.unique(labels_slic)

    rows = []
    for sp_id in unique_sp:
        mask = labels_slic == sp_id
        if not np.any(mask):
            continue

        # Médias e desvios padrão de cor
        rgb_region = img_rgb_f[mask]
        hsv_region = img_hsv[mask]
        lab_region = img_lab[mask]

        rgb_means = rgb_region.mean(axis=0)
        hsv_means = hsv_region.mean(axis=0)
        lab_means = lab_region.mean(axis=0)

        rgb_stds = rgb_region.std(axis=0)
        hsv_stds = hsv_region.std(axis=0)
        lab_stds = lab_region.std(axis=0)

        # Recorte para textura (bbox da máscara)
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        sub_gray_q = img_gray_q[y0:y1, x0:x1]

        glcm_feats = _compute_glcm_features(sub_gray_q)

        row = {
            'superpixel_id': int(sp_id),
            'rgb_mean_ch1': float(rgb_means[0]),
            'rgb_mean_ch2': float(rgb_means[1]),
            'rgb_mean_ch3': float(rgb_means[2]),
            'rgb_std_ch1': float(rgb_stds[0]),
            'rgb_std_ch2': float(rgb_stds[1]),
            'rgb_std_ch3': float(rgb_stds[2]),
            'hsv_mean_ch1': float(hsv_means[0]),
            'hsv_mean_ch2': float(hsv_means[1]),
            'hsv_mean_ch3': float(hsv_means[2]),
            'hsv_std_ch1': float(hsv_stds[0]),
            'hsv_std_ch2': float(hsv_stds[1]),
            'hsv_std_ch3': float(hsv_stds[2]),
            'lab_mean_ch1': float(lab_means[0]),
            'lab_mean_ch2': float(lab_means[1]),
            'lab_mean_ch3': float(lab_means[2]),
            'lab_std_ch1': float(lab_stds[0]),
            'lab_std_ch2': float(lab_stds[1]),
            'lab_std_ch3': float(lab_stds[2]),
            **glcm_feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ordena por id para estabilidade
    if not df.empty:
        df = df.sort_values('superpixel_id').reset_index(drop=True)
    return df


