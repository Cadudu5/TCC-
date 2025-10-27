import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image


# --- CONFIGURAÇÕES ---
IMAGE_PATH = ''  # inicia sem imagem; escolha após abrir com a tecla 'o'
N_SEGMENTS = 5000
COMPACTNESS = 10
SIGMA = 3


def extract_features(image, superpixels):
    """
    Calcula características de cor e textura para cada superpixel.
    Mantém o mesmo conjunto de features do rotulador original.
    """
    print("Iniciando extração de características. Isso pode demorar...")

    hsv_image = rgb2hsv(image)
    lab_image = rgb2lab(image)
    gray_image = rgb2gray(image)
    gray_image_uint8 = (gray_image * 255).astype('uint8')

    unique_superpixels = np.unique(superpixels)
    all_features = []

    for superpixel_id in tqdm(unique_superpixels, desc="Extraindo Características"):
        mask = (superpixels == superpixel_id)
        features = {'superpixel_id': int(superpixel_id)}

        # Cor (médias e desvios para RGB/HSV/LAB)
        color_spaces = {
            'rgb': image,
            'hsv': hsv_image,
            'lab': lab_image,
        }
        for name, img_space in color_spaces.items():
            for channel in range(img_space.shape[2]):
                channel_pixels = img_space[:, :, channel][mask]
                features[f'{name}_mean_ch{channel+1}'] = float(np.mean(channel_pixels))
                features[f'{name}_std_ch{channel+1}'] = float(np.std(channel_pixels))

        # Textura (GLCM) usando ROI do bounding box
        rows, cols = np.where(mask)
        min_row, max_row = int(np.min(rows)), int(np.max(rows))
        min_col, max_col = int(np.min(cols)), int(np.max(cols))
        roi = gray_image_uint8[min_row:max_row+1, min_col:max_col+1]

        glcm = graycomatrix(
            roi,
            distances=[1, 3, 5],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True,
        )
        texture_props = ['contrast', 'dissimilarity', 'homogeneity', 'correlation']
        for prop in texture_props:
            features[f'glcm_{prop}'] = float(np.mean(graycoprops(glcm, prop)))

        all_features.append(features)

    print("Extração de características concluída.")
    return pd.DataFrame(all_features)


class SuperpixelBackgroundLabelerLiteGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rotulador Lite de Fundo de Superpixels")

        # Estado
        self.image_path = None
        self.original_image = None
        self.superpixels = None
        self.labels_df = None
        self.color_mask = None

        # --- Interface Gráfica (Widgets) ---
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.btn_load = tk.Button(self.top_frame, text="Carregar Imagem", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT)

        self.btn_save = tk.Button(self.top_frame, text="Salvar Rótulos", command=self.save_results, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.main_frame, text="Carregue uma imagem para começar.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Canvas do Matplotlib ---
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Eventos do mouse no canvas
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Pan state e artista da máscara
        self._is_panning = False
        self._pan_start_x = None
        self._pan_start_y = None
        self.mask_artist = None

        # Pintura (multiseleção) por arraste
        self._is_painting = False
        self._moved_since_press = False
        self._paint_label_value = 1  # 1=Fundo, 0=Não-Fundo
        self._painted_ids_in_stroke = set()

    def update_status(self, message: str):
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
            # Usa PIL para lidar melhor com .tiff (inclusive multi-página): pega a 1ª página e converte para RGB
            img = Image.open(self.image_path).convert('RGB')
            self.original_image = img_as_float(np.array(img))

            self.update_status("Calculando superpixels... Isso pode levar um momento.")
            self.superpixels = slic(
                self.original_image,
                n_segments=N_SEGMENTS,
                compactness=COMPACTNESS,
                sigma=SIGMA,
                start_label=1,
            )

            # Cria DataFrame de rótulos (1=Fundo, 0=Não-Fundo)
            unique_superpixels = np.unique(self.superpixels)
            self.labels_df = pd.DataFrame({'superpixel_id': unique_superpixels, 'label': 0})

            # Máscara RGBA vermelha para fundo
            self.color_mask = np.zeros((*self.original_image.shape[:2], 4), dtype=float)
            self.update_visualization()
            self.btn_save.config(state=tk.NORMAL)
            self.update_status("Pronto! Scroll=Zoom | Botão Direito+Arrastar=Mover | Botão Meio=Resetar Zoom | Botão Esquerdo=Rotular FUNDO.")

        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar ou processar a imagem.\n\n{e}")
            self.update_status("Erro ao carregar a imagem.")

    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            # Inicia pintura: Shift+Esquerdo => Não-Fundo(0); Esquerdo => Fundo(1)
            if self.superpixels is None:
                return
            key = getattr(event, 'key', None)
            shift_pressed = False
            if isinstance(key, str):
                shift_pressed = ('shift' in key.lower())
            self._paint_label_value = 0 if shift_pressed else 1
            self._is_painting = True
            self._moved_since_press = False
            self._painted_ids_in_stroke.clear()
            # Aplica imediatamente no ponto inicial
            try:
                x, y = int(event.xdata), int(event.ydata)
                self._paint_at(x, y)
            except (TypeError, ValueError):
                pass
        elif event.button == 2:
            self.reset_view()
        elif event.button == 3:
            self._is_panning = True
            self._pan_start_x = event.x
            self._pan_start_y = event.y

    def on_mouse_release(self, event):
        if event.button == 3:
            self._is_panning = False
            self._pan_start_x = None
            self._pan_start_y = None
        elif event.button == 1:
            # Se não houve movimento, trate como clique único (toggle)
            if not self._moved_since_press and self.superpixels is not None:
                self._toggle_label_at_event(event)
            self._is_painting = False
            self._painted_ids_in_stroke.clear()

    def on_click_label(self, event):
        try:
            x, y = int(event.xdata), int(event.ydata)
            h, w, _ = self.original_image.shape
            if not (0 <= y < h and 0 <= x < w):
                return
        except (ValueError, TypeError):
            return

        clicked_id = self.superpixels[y, x]

        # Alterna o rótulo (1=Fundo, 0=Não-Fundo)
        current_label = self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'].iloc[0]
        new_label = 1 - current_label
        self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'] = new_label

        mask = (self.superpixels == clicked_id)
        if new_label == 1:
            self.update_status(f"Superpixel {clicked_id} marcado como FUNDO.")
            self.color_mask[mask] = [1, 0, 0, 0.5]  # vermelho para fundo
        else:
            self.update_status(f"Superpixel {clicked_id} desmarcado (NÃO-FUNDO).")
            self.color_mask[mask] = [0, 0, 0, 0]    # transparente

        if self.mask_artist is not None:
            self.mask_artist.set_data(self.color_mask)
        self.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale_factor = 1.1 if event.step > 0 else 1 / 1.1
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata
        new_xlim = [x - (x - xlim[0]) / scale_factor, x + (xlim[1] - x) / scale_factor]
        new_ylim = [y - (y - ylim[0]) / scale_factor, y + (ylim[1] - y) / scale_factor]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        # Pan com botão direito
        if self._is_panning:
            dx = event.x - self._pan_start_x
            dy = event.y - self._pan_start_y
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()
            self._pan_start_x = event.x
            self._pan_start_y = event.y
            return
        # Pintura por arraste com botão esquerdo
        if self._is_painting and self.superpixels is not None:
            try:
                x, y = int(event.xdata), int(event.ydata)
            except (TypeError, ValueError):
                return
            self._moved_since_press = True
            self._paint_at(x, y)

    def _toggle_label_at_event(self, event):
        try:
            x, y = int(event.xdata), int(event.ydata)
            h, w, _ = self.original_image.shape
            if not (0 <= y < h and 0 <= x < w):
                return
        except (ValueError, TypeError):
            return
        clicked_id = self.superpixels[y, x]
        current_label = self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'].iloc[0]
        new_label = 1 - current_label
        self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'] = new_label
        mask = (self.superpixels == clicked_id)
        if new_label == 1:
            self.update_status(f"Superpixel {clicked_id} marcado como FUNDO.")
            self.color_mask[mask] = [1, 0, 0, 0.5]
        else:
            self.update_status(f"Superpixel {clicked_id} desmarcado (NÃO-FUNDO).")
            self.color_mask[mask] = [0, 0, 0, 0]
        if self.mask_artist is not None:
            self.mask_artist.set_data(self.color_mask)
        self.canvas.draw_idle()

    def _paint_at(self, x: int, y: int):
        h, w, _ = self.original_image.shape
        if not (0 <= y < h and 0 <= x < w):
            return
        clicked_id = int(self.superpixels[y, x])
        if clicked_id in self._painted_ids_in_stroke:
            return
        self._painted_ids_in_stroke.add(clicked_id)
        current_label = int(self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'].iloc[0])
        if current_label == self._paint_label_value:
            return
        # Aplica valor de pintura
        self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'] = int(self._paint_label_value)
        mask = (self.superpixels == clicked_id)
        if self._paint_label_value == 1:
            self.color_mask[mask] = [1, 0, 0, 0.5]
        else:
            self.color_mask[mask] = [0, 0, 0, 0]
        if self.mask_artist is not None:
            self.mask_artist.set_data(self.color_mask)
        self.canvas.draw_idle()

    def reset_view(self):
        if self.original_image is not None:
            h, w, _ = self.original_image.shape
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
            self.canvas.draw_idle()

    def update_visualization(self):
        if self.original_image is None:
            return
        self.ax.clear()
        self.ax.imshow(mark_boundaries(self.original_image, self.superpixels, color=(0, 0, 0)))
        self.mask_artist = self.ax.imshow(self.color_mask)
        self.ax.set_title("Clique Esquerdo: Marcar/Desmarcar FUNDO | Botão Meio: Resetar Zoom")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.reset_view()
        self.canvas.draw()

    def save_results(self):
        if self.labels_df is None or self.image_path is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para salvar.")
            return
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        csv_path = filedialog.asksaveasfilename(
            title="Salvar CSV com Rótulos de Fundo",
            initialfile=f"rotulos_fundo_{base_name}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if csv_path:
            try:
                self.labels_df.to_csv(csv_path, index=False)
                self.update_status(f"CSV de rótulos de fundo salvo em {os.path.basename(csv_path)}")
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo CSV.\n\n{e}")
                return

        image_save_path = filedialog.asksaveasfilename(
            title="Salvar imagem com marcações",
            initialfile=f"visualizacao_fundo_{base_name}.png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        if image_save_path:
            try:
                self.fig.savefig(image_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
                self.update_status("CSV e imagem de visualização salvos com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar a imagem.\n\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = SuperpixelBackgroundLabelerLiteGUI(root)
    root.geometry("800x800")
    root.mainloop()


