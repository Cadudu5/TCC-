import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm # Importa a barra de progresso

# --- CONFIGURAÇÕES ---
IMAGE_PATH = 'imagem5.tif' 
OUTPUT_CSV = 'dataset_final_imagem5.csv'
N_SEGMENTS = 5000
COMPACTNESS = 10
SIGMA = 3

def extract_features(image, superpixels):
    """
    Calcula características de cor e textura para cada superpixel.
    """
    print("Iniciando extração de características. Isso pode demorar...")
    
    # Converte a imagem para outros espaços de cor uma única vez para eficiência
    hsv_image = rgb2hsv(image)
    lab_image = rgb2lab(image)
    gray_image = rgb2gray(image)
    # Converte imagem cinza para o range de 0-255 (necessário para GLCM)
    gray_image_uint8 = (gray_image * 255).astype('uint8')
    
    unique_superpixels = np.unique(superpixels)
    all_features = []

    # Usa tqdm para mostrar uma barra de progresso no terminal
    for superpixel_id in tqdm(unique_superpixels, desc="Extraindo Características"):
        # Cria máscara para o superpixel atual
        mask = (superpixels == superpixel_id)
        
        # Dicionário para armazenar as características do superpixel atual
        features = {'superpixel_id': superpixel_id}
        
        # --- 1. Características de Cor ---
        color_spaces = {
            'rgb': image,
            'hsv': hsv_image,
            'lab': lab_image
        }
        
        for name, img_space in color_spaces.items():
            for channel in range(img_space.shape[2]):
                channel_pixels = img_space[mask, channel]
                features[f'{name}_mean_ch{channel+1}'] = np.mean(channel_pixels)
                features[f'{name}_std_ch{channel+1}'] = np.std(channel_pixels)

        # --- 2. Características de Textura (GLCM) ---
        # Encontra a "caixa" (bounding box) que contém o superpixel para otimizar
        rows, cols = np.where(mask)
        min_row, max_row, min_col, max_col = min(rows), max(rows), min(cols), max(cols)
        
        # Extrai a região da imagem em escala de cinza
        roi = gray_image_uint8[min_row:max_row+1, min_col:max_col+1]
        
        # Calcula GLCM. Distâncias e ângulos comuns para robustez.
        glcm = graycomatrix(roi, distances=[1, 3, 5], 
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256,
                            symmetric=True, normed=True)
        
        # Calcula propriedades da GLCM e tira a média para ter um valor por propriedade
        texture_props = ['contrast', 'dissimilarity', 'homogeneity', 'correlation']
        for prop in texture_props:
            features[f'glcm_{prop}'] = np.mean(graycoprops(glcm, prop))
            
        all_features.append(features)
        
    print("Extração de características concluída.")
    return pd.DataFrame(all_features)


class SuperpixelLabeler:
    def __init__(self, image_path):
        try:
            self.original_image = img_as_float(imread(image_path))
        except FileNotFoundError:
            print(f"Erro: Arquivo de imagem não encontrado em '{image_path}'"); exit()
            
        print("Calculando superpixels...")
        self.superpixels = slic(self.original_image, n_segments=N_SEGMENTS, compactness=COMPACTNESS, sigma=SIGMA, start_label=1)
        # Salva o mapa de superpixels para uso futuro
        self.superpixel_map = self.superpixels.copy()
        # --- CORREÇÃO E MELHORIA ---
        # 1. Cria um nome de arquivo de saída dinâmico a partir do nome do arquivo de entrada
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        map_save_path = f'mapa_superpixel/mapa_superpixels_{base_name}.npy'
        
        # 2. Salva o atributo correto da classe (self.superpixels)
        np.save(map_save_path, self.superpixels)
        print(f"✅ Mapa de superpixels salvo em: '{map_save_path}'")

        # EXTRAI AS CARACTERÍSTICAS E ARMAZENA NUM DATAFRAME
        self.features_df = extract_features(self.original_image, self.superpixels)
        # ADICIONA A COLUNA DE RÓTULOS, INICIALMENTE COM 0
        self.features_df['label'] = 0
        
        self.color_mask = np.zeros((*self.original_image.shape[:2], 4), dtype=float)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def setup_plot(self):
        self.ax.clear()
        self.ax.imshow(mark_boundaries(self.original_image, self.superpixels))
        self.ax.imshow(self.color_mask)
        self.ax.set_title("Rotulador de Superpixels\n"
                          "Clique Esquerdo: POSITIVO | Clique Direito: NEGATIVO\n"
                          "Pressione 's' para Salvar | 'q' para Sair")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def update_visualization(self, superpixel_id):
        mask = (self.superpixels == superpixel_id)
        # Pega o rótulo atual do DataFrame
        label_status = self.features_df.loc[self.features_df['superpixel_id'] == superpixel_id, 'label'].iloc[0]
        
        if label_status == 1:
            self.color_mask[mask] = [0, 1, 0, 0.5]  # Verde
        else:
            self.color_mask[mask] = [0, 0, 0, 0]    # Transparente
        self.setup_plot()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        x, y = int(event.xdata), int(event.ydata)
        clicked_id = self.superpixels[y, x]
        
        # Atualiza o DataFrame do pandas
        if event.button == 1:
            self.features_df.loc[self.features_df['superpixel_id'] == clicked_id, 'label'] = 1
            print(f"Superpixel {clicked_id} marcado como POSITIVO (Neutrófilo).")
        elif event.button == 3:
            self.features_df.loc[self.features_df['superpixel_id'] == clicked_id, 'label'] = 0
            print(f"Superpixel {clicked_id} marcado como NEGATIVO.")
        
        self.update_visualization(clicked_id)

    def on_key_press(self, event):
        if event.key == 's': self.save_data()
        elif event.key == 'q': plt.close(self.fig); print("Janela fechada.")

    def on_close(self, event):
        self.save_data()
        print("Janela fechada. Dados salvos automaticamente.")

    def save_data(self):
        """Salva o DataFrame completo (características + rótulos) em CSV."""
        self.features_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n>>> Dataset completo salvo em '{OUTPUT_CSV}'! <<<\n")

    def run(self):
        print("-" * 50); print("Iniciando a ferramenta. Aguarde a janela abrir."); print("-" * 50)
        plt.show()

if __name__ == '__main__':
    if not os.path.exists(IMAGE_PATH):
        print(f"ERRO: A imagem '{IMAGE_PATH}' não foi encontrada.")
    else:
        labeler = SuperpixelLabeler(IMAGE_PATH)
        labeler.run()