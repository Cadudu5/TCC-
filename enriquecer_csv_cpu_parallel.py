import os
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
# 1. Pastas de Entrada e Saída
# Ajuste estes caminhos conforme sua organização de arquivos.
LABELS_DIR = 'padrao_ouro'           # Pasta alvo contendo os CSVs de rótulos
IMAGES_DIR = 'Neutrofilos'           # Pasta contendo as imagens correspondentes
OUTPUT_DIR = 'imagens_enriquecidas'  # Pasta destino para salvar os CSVs enriquecidos

# 2. Parâmetros do Superpixel (MUITO IMPORTANTE!)
# ESTES VALORES DEVEM SER EXATAMENTE OS MESMOS USADOS NO SCRIPT DE ROTULAÇÃO ORIGINAL.
N_SEGMENTS = 5000
COMPACTNESS = 10
SIGMA = 3

# 3. Paralelismo (CPU)
# None => usa os.cpu_count(); ajuste para limitar threads.
MAX_WORKERS = None


def extract_features_parallel(image, superpixels, max_workers=None):
    """
    Calcula características de cor e textura para cada superpixel em paralelo (CPU threads).
    Mantém o mesmo conjunto de características do script original.
    """
    print("Iniciando extração de características (CPU paralela). Isso pode demorar...")

    hsv_image = rgb2hsv(image)
    lab_image = rgb2lab(image)
    gray_image_uint8 = (rgb2gray(image) * 255).astype('uint8')

    unique_superpixels = np.unique(superpixels)

    color_spaces = {
        'rgb': image,
        'hsv': hsv_image,
        'lab': lab_image,
    }

    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    texture_props = ['contrast', 'dissimilarity', 'homogeneity', 'correlation']

    def compute_features(superpixel_id):
        mask = (superpixels == superpixel_id)

        # Pode ocorrer de um rótulo estar ausente em casos raros; proteger.
        if not np.any(mask):
            return {'superpixel_id': int(superpixel_id)}

        features = {'superpixel_id': int(superpixel_id)}

        # Cor
        for name, img_space in color_spaces.items():
            # img_space shape: (H, W, C)
            for channel in range(img_space.shape[2]):
                channel_pixels = img_space[:, :, channel][mask]
                # Se o superpixel tiver 1 único pixel, std será 0.0
                features[f'{name}_mean_ch{channel+1}'] = float(np.mean(channel_pixels))
                features[f'{name}_std_ch{channel+1}'] = float(np.std(channel_pixels))

        # Textura (GLCM) com recorte de ROI
        rows, cols = np.where(mask)
        r0, r1 = int(np.min(rows)), int(np.max(rows))
        c0, c1 = int(np.min(cols)), int(np.max(cols))
        roi = gray_image_uint8[r0:r1+1, c0:c1+1]

        # Se ROI for vazia por algum motivo inesperado, pula textura
        if roi.size > 0:
            glcm = graycomatrix(
                roi,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True,
            )
            for prop in texture_props:
                features[f'glcm_{prop}'] = float(np.mean(graycoprops(glcm, prop)))

        return features

    all_features = []
    workers = max_workers if max_workers is not None else os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for feat in tqdm(
            executor.map(compute_features, unique_superpixels),
            total=len(unique_superpixels),
            desc="Extraindo Características (CPU paralelo)"):
            all_features.append(feat)

    print("Extração de características concluída.")
    return pd.DataFrame(all_features)


def main():
    """
    Função principal para executar o processo de enriquecimento em lote dos CSVs com paralelismo de CPU.
    """
    print("--- Iniciando enriquecimento em lote de CSVs (CPU paralela) ---")

    # --- 0. Validações e preparo ---
    if not os.path.isdir(LABELS_DIR):
        print(f"ERRO: Pasta de rótulos não encontrada: '{LABELS_DIR}'")
        return
    if not os.path.isdir(IMAGES_DIR):
        print(f"ERRO: Pasta de imagens não encontrada: '{IMAGES_DIR}'")
        return
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lista de arquivos CSV de rótulos
    csv_files = [f for f in os.listdir(LABELS_DIR) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"Nenhum arquivo CSV encontrado em '{LABELS_DIR}'. Nada a processar.")
        return

    # Mapa de arquivos de imagem disponíveis (base do nome -> caminho)
    allowed_exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    image_name_to_path = {}
    for fname in os.listdir(IMAGES_DIR):
        name, ext = os.path.splitext(fname)
        if ext.lower() in allowed_exts:
            image_name_to_path[name.lower()] = os.path.join(IMAGES_DIR, fname)

    def candidate_basenames(csv_basename):
        """
        Gera candidatos de nomes-base de imagem a partir do nome do CSV de rótulos.
        Exemplos:
        - rotulos_7 dias 40x -> 7 dias 40x
        - rotulos_superpixels_imagem1 -> imagem1
        - 7 dias 40x -> 7 dias 40x
        """
        bases = []
        base = csv_basename
        bases.append(base)
        if base.lower().startswith('rotulos_superpixels_'):
            bases.append(base[len('rotulos_superpixels_'):])
        if base.lower().startswith('rotulos_'):
            bases.append(base[len('rotulos_'):])
        # Remover duplicatas preservando ordem
        seen = set()
        ordered = []
        for b in bases:
            bl = b.lower()
            if bl not in seen:
                ordered.append(b)
                seen.add(bl)
        return ordered

    processados = 0
    pulados = 0

    for csv_name in tqdm(sorted(csv_files), desc="Processando arquivos"):
        labels_csv_path = os.path.join(LABELS_DIR, csv_name)
        csv_basename = os.path.splitext(csv_name)[0]

        # Encontrar imagem correspondente por nome
        image_path = None
        for cand in candidate_basenames(csv_basename):
            found = image_name_to_path.get(cand.lower())
            if found is not None:
                image_path = found
                break

        if image_path is None:
            print(f"AVISO: Imagem correspondente não encontrada para '{csv_name}'. Pulando.")
            pulados += 1
            continue

        try:
            # 1) Carregar imagem e recalcular superpixels
            print(f"\nCarregando imagem: {image_path}")
            image = img_as_float(imread(image_path))

            print("Recalculando superpixels (parâmetros idênticos ao script de rotulação)...")
            superpixels = slic(
                image,
                n_segments=N_SEGMENTS,
                compactness=COMPACTNESS,
                sigma=SIGMA,
                start_label=1,
            )

            # 2) Extrair características (paralelo em CPU)
            features_df = extract_features_parallel(image, superpixels, max_workers=MAX_WORKERS)

            # 3) Carregar rótulos
            print(f"Carregando rótulos: {labels_csv_path}")
            labels_df = pd.read_csv(labels_csv_path)

            # 4) Juntar características e rótulos
            print("Juntando características com rótulos...")
            final_df = pd.merge(features_df, labels_df, on='superpixel_id', how='left')
            final_df['label'] = final_df['label'].fillna(0).astype(int)

            cols = list(final_df.columns)
            if 'label' in cols:
                cols.remove('label')
                final_df = final_df[cols + ['label']]

            # 5) Salvar no destino com sufixo _enriquecido
            out_name = f"{csv_basename}_enriquecido.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            final_df.to_csv(out_path, index=False)

            print(f"Salvo: '{out_path}'  | Linhas: {final_df.shape[0]}  Colunas: {final_df.shape[1]}")
            processados += 1

        except Exception as exc:
            print(f"ERRO ao processar '{csv_name}': {exc}")
            pulados += 1
            continue

    print("\n--- Enriquecimento em lote concluído (CPU paralela) ---")
    print(f"Processados com sucesso: {processados}")
    print(f"Pulados/Com erro: {pulados}")


if __name__ == '__main__':
    main()


