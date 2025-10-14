import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- CONFIGURAÇÕES GLOBAIS ---
N_SEGMENTS = 5000
COMPACTNESS = 10
SIGMA = 3

def extract_features(image, superpixels, progress_callback=None):
    """
    Calcula características de cor (média) e textura para cada superpixel.
    Apenas as médias dos canais de cor são extraídas, sem o desvio padrão.
    """
    hsv_image = rgb2hsv(image)
    lab_image = rgb2lab(image)
    gray_image = rgb2gray(image)
    gray_image_uint8 = (gray_image * 255).astype('uint8')
    
    unique_superpixels = np.unique(superpixels)
    all_features = []

    total_superpixels = len(unique_superpixels)
    for i, superpixel_id in enumerate(unique_superpixels):
        if progress_callback:
            progress_callback(f"Extraindo características: Superpixel {i+1}/{total_superpixels}")

        mask = (superpixels == superpixel_id)
        features = {'superpixel_id': superpixel_id}
        
        # --- 1. Características de Cor (Apenas Média) ---
        color_spaces = {'rgb': image, 'hsv': hsv_image, 'lab': lab_image}
        for name, img_space in color_spaces.items():
            for channel in range(img_space.shape[2]):
                channel_pixels = img_space[mask, channel]
                features[f'{name}_mean_ch{channel+1}'] = np.mean(channel_pixels)

        # --- 2. Características de Textura (GLCM) ---
        rows, cols = np.where(mask)
        min_row, max_row, min_col, max_col = min(rows), max(rows), min(cols), max(cols)
        roi = gray_image_uint8[min_row:max_row+1, min_col:max_col+1]
        
        glcm = graycomatrix(roi, distances=[1, 3, 5], 
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        
        texture_props = ['contrast', 'dissimilarity', 'homogeneity', 'correlation']
        for prop in texture_props:
            features[f'glcm_{prop}'] = np.mean(graycoprops(glcm, prop))
            
        all_features.append(features)
        
    return pd.DataFrame(all_features)

class SuperpixelLabelerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rotulador de Superpixels para Neutrófilos")
        
        # Variáveis de estado
        self.image_path = None
        self.original_image = None
        self.superpixels = None
        self.features_df = None
        self.color_mask = None

        # --- Interface Gráfica (Widgets) ---
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.btn_load = tk.Button(self.top_frame, text="Carregar Imagem", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT)

        self.btn_save = tk.Button(self.top_frame, text="Salvar Resultados", command=self.save_results, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.main_frame, text="Carregue uma imagem para começar.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Canvas do Matplotlib para a imagem ---
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.tif *.tiff *.png *.jpg *.jpeg"), ("Todos os arquivos", "*.*")]
        )
        if not path:
            return

        self.image_path = path
        self.update_status(f"Carregando imagem: {os.path.basename(self.image_path)}...")

        try:
            self.original_image = img_as_float(imread(self.image_path))
            if self.original_image.shape[2] == 4:
                self.original_image = self.original_image[:, :, :3]

            self.update_status("Calculando superpixels... Isso pode levar um momento.")
            self.superpixels = slic(self.original_image, n_segments=N_SEGMENTS, compactness=COMPACTNESS, sigma=SIGMA, start_label=1)
            
            self.update_status("Extraindo características dos superpixels...")
            self.features_df = extract_features(self.original_image, self.superpixels, self.update_status)
            self.features_df['label'] = 0  # 0 para negativo, 1 para positivo

            self.color_mask = np.zeros((*self.original_image.shape[:2], 4), dtype=float)
            self.update_visualization()
            self.btn_save.config(state=tk.NORMAL)
            self.update_status("Imagem pronta. Clique nos superpixels para rotular como 'Positivo' (Neutrófilo).")

        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar ou processar a imagem.\n\n{e}")
            self.update_status("Erro ao carregar a imagem.")

    def on_click(self, event):
        if not all([event.inaxes == self.ax, event.xdata, event.ydata, self.superpixels is not None]):
            return

        x, y = int(event.xdata), int(event.ydata)
        h, w, _ = self.original_image.shape
        if not (0 <= y < h and 0 <= x < w):
            return

        clicked_id = self.superpixels[y, x]
        
        # Alterna o rótulo: se 1 vira 0, se 0 vira 1
        current_label = self.features_df.loc[self.features_df['superpixel_id'] == clicked_id, 'label'].iloc[0]
        new_label = 1 - current_label

        self.features_df.loc[self.features_df['superpixel_id'] == clicked_id, 'label'] = new_label

        mask = (self.superpixels == clicked_id)
        if new_label == 1:
            self.update_status(f"Superpixel {clicked_id} marcado como POSITIVO.")
            self.color_mask[mask] = [0, 1, 0, 0.5]  # Verde para positivo
        else:
            self.update_status(f"Superpixel {clicked_id} desmarcado (NEGATIVO).")
            self.color_mask[mask] = [0, 0, 0, 0]    # Transparente para negativo

        self.update_visualization()

    def update_visualization(self):
        if self.original_image is None:
            return
            
        self.ax.clear()
        self.ax.imshow(mark_boundaries(self.original_image, self.superpixels, color=(0, 0, 0)))
        self.ax.imshow(self.color_mask)
        self.ax.set_title("Clique para marcar/desmarcar superpixels (Neutrófilos)")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw()

    def save_results(self):
        if self.features_df is None or self.image_path is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para salvar.")
            return

        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Salvar CSV
        csv_path = filedialog.asksaveasfilename(
            title="Salvar CSV com características",
            initialfile=f"dataset_{base_name}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if csv_path:
            try:
                self.features_df.to_csv(csv_path, index=False)
                self.update_status(f"CSV salvo em {os.path.basename(csv_path)}")
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo CSV.\n\n{e}")
                return

        # Salvar Imagem
        image_save_path = filedialog.asksaveasfilename(
            title="Salvar imagem com marcações",
            initialfile=f"rotulado_{base_name}.png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        if image_save_path:
            try:
                self.fig.savefig(image_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
                self.update_status("CSV e Imagem salvos com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar a imagem.\n\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SuperpixelLabelerGUI(root)
    root.geometry("800x800")
    root.mainloop()
