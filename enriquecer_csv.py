import os
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
# 1. Arquivos de Entrada e Saída
IMAGE_PATH = '7 dias 40x.tif'
LABELS_CSV_PATH = 'rotulos_superpixels.csv'  # SEU CSV JÁ ROTULADO
OUTPUT_CSV_PATH = 'dataset_final_com_features.csv' # Arquivo final que será criado

# 2. Parâmetros do Superpixel (MUITO IMPORTANTE!)
# ESTES VALORES DEVEM SER EXATAMENTE OS MESMOS USADOS NO SCRIPT DE ROTULAÇÃO ORIGINAL.
N_SEGMENTS = 5000
COMPACTNESS = 10
SIGMA = 3

def extract_features(image, superpixels):
    """
    Calcula características de cor e textura para cada superpixel.
    (Esta função é a mesma do script anterior)
    """
    print("Iniciando extração de características. Isso pode demorar...")
    
    hsv_image = rgb2hsv(image)
    lab_image = rgb2lab(image)
    gray_image_uint8 = (rgb2gray(image) * 255).astype('uint8')
    
    unique_superpixels = np.unique(superpixels)
    all_features = []

    for superpixel_id in tqdm(unique_superpixels, desc="Extraindo Características"):
        mask = (superpixels == superpixel_id)
        features = {'superpixel_id': superpixel_id}
        
        # Cor
        color_spaces = {'rgb': image, 'hsv': hsv_image, 'lab': lab_image}
        for name, img_space in color_spaces.items():
            for channel in range(img_space.shape[2]):
                channel_pixels = img_space[mask, channel]
                features[f'{name}_mean_ch{channel+1}'] = np.mean(channel_pixels)
                features[f'{name}_std_ch{channel+1}'] = np.std(channel_pixels)

        # Textura (GLCM)
        rows, cols = np.where(mask)
        roi = gray_image_uint8[min(rows):max(rows)+1, min(cols):max(cols)+1]
        
        glcm = graycomatrix(roi, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        
        texture_props = ['contrast', 'dissimilarity', 'homogeneity', 'correlation']
        for prop in texture_props:
            features[f'glcm_{prop}'] = np.mean(graycoprops(glcm, prop))
            
        all_features.append(features)
        
    print("Extração de características concluída.")
    return pd.DataFrame(all_features)

def main():
    """
    Função principal para executar o processo de enriquecimento do CSV.
    """
    print("--- Iniciando Script para Enriquecer CSV ---")

    # --- 1. Validação dos arquivos de entrada ---
    if not os.path.exists(IMAGE_PATH):
        print(f"ERRO: Arquivo de imagem não encontrado em '{IMAGE_PATH}'")
        return
    if not os.path.exists(LABELS_CSV_PATH):
        print(f"ERRO: Arquivo de rótulos não encontrado em '{LABELS_CSV_PATH}'")
        return

    # --- 2. Carregar imagem e recalcular superpixels ---
    print(f"Carregando a imagem: {IMAGE_PATH}")
    image = img_as_float(imread(IMAGE_PATH))
    
    print("Recalculando os superpixels (garanta que os parâmetros são os mesmos)...")
    superpixels = slic(image, n_segments=N_SEGMENTS, compactness=COMPACTNESS, sigma=SIGMA, start_label=1)

    # --- 3. Extrair características ---
    features_df = extract_features(image, superpixels)

    # --- 4. Carregar rótulos ---
    print(f"Carregando os rótulos do arquivo: {LABELS_CSV_PATH}")
    labels_df = pd.read_csv(LABELS_CSV_PATH)

    # --- 5. Juntar características e rótulos ---
    print("Juntando as características calculadas com os rótulos manuais...")
    # Usa 'merge' do pandas para combinar os dois DataFrames com base na coluna 'superpixel_id'
    final_df = pd.merge(features_df, labels_df, on='superpixel_id', how='left')

    # Garante que qualquer superpixel que não foi rotulado (ficando com valor NaN) receba o rótulo 0 (negativo)
    final_df['label'] = final_df['label'].fillna(0).astype(int)
    
    # (Opcional) Reordena as colunas para deixar 'label' no final
    cols = list(final_df.columns)
    cols.remove('label')
    final_df = final_df[cols + ['label']]

    # --- 6. Salvar o dataset final ---
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n--- Processo Concluído com Sucesso! ---")
    print(f"Dataset final salvo em: '{OUTPUT_CSV_PATH}'")
    print(f"O dataset final contém {final_df.shape[0]} linhas (superpixels) e {final_df.shape[1]} colunas (ID + características + rótulo).")


if __name__ == '__main__':
    main()