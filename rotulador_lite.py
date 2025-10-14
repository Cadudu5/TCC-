import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import pandas as pd
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from skimage.util import img_as_float
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- CONFIGURAÇÕES GLOBAIS DE SEGMENTAÇÃO ---
# É crucial que estes parâmetros sejam os mesmos no script de processamento final
N_SEGMENTS = 5000
COMPACTNESS = 10
SIGMA = 3

class SuperpixelLabelerLiteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rotulador Lite de Superpixels")
        
        # Variáveis de estado
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

        # --- Canvas do Matplotlib para a imagem ---
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Conecta todos os eventos do mouse
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Variáveis para o Pan e para o artista da máscara
        self._is_panning = False
        self._pan_start_x = None
        self._pan_start_y = None
        self.mask_artist = None

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
            if self.original_image.ndim == 2:
                 # Se for escala de cinza, converte para RGB
                self.original_image = np.stack((self.original_image,) * 3, axis=-1)
            if self.original_image.shape[2] == 4: # Remove canal alfa se existir
                self.original_image = self.original_image[:, :, :3]

            self.update_status("Calculando superpixels... Isso pode levar um momento.")
            self.superpixels = slic(self.original_image, n_segments=N_SEGMENTS, compactness=COMPACTNESS, sigma=SIGMA, start_label=1)
            
            # Cria um DataFrame simples apenas com IDs e rótulos
            unique_superpixels = np.unique(self.superpixels)
            self.labels_df = pd.DataFrame({'superpixel_id': unique_superpixels, 'label': 0})

            self.color_mask = np.zeros((*self.original_image.shape[:2], 4), dtype=float)
            self.update_visualization()
            self.btn_save.config(state=tk.NORMAL)
            self.update_status("Pronto! Scroll=Zoom | Botão Direito+Arrastar=Mover | Botão Meio=Resetar Zoom | Botão Esquerdo=Rotular.")

        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar ou processar a imagem.\n\n{e}")
            self.update_status("Erro ao carregar a imagem.")

    def on_mouse_press(self, event):
        if not all([event.inaxes == self.ax, self.superpixels is not None]):
            return

        # Botão Esquerdo: Rotulagem
        if event.button == 1:
            self.on_click_label(event)
        # Botão do Meio: Resetar a visualização
        elif event.button == 2:
            self.reset_view()
        # Botão Direito: Iniciar Pan
        elif event.button == 3:
            self._is_panning = True
            self._pan_start_x = event.x
            self._pan_start_y = event.y

    def on_mouse_release(self, event):
        if event.button == 3:
            self._is_panning = False
            self._pan_start_x = None
            self._pan_start_y = None

    def on_click_label(self, event):
        try:
            x, y = int(event.xdata), int(event.ydata)
            h, w, _ = self.original_image.shape
            if not (0 <= y < h and 0 <= x < w):
                return
        except (ValueError, TypeError):
            return

        clicked_id = self.superpixels[y, x]
        
        # Alterna o rótulo
        current_label = self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'].iloc[0]
        new_label = 1 - current_label
        self.labels_df.loc[self.labels_df['superpixel_id'] == clicked_id, 'label'] = new_label

        mask = (self.superpixels == clicked_id)
        if new_label == 1:
            self.update_status(f"Superpixel {clicked_id} marcado como POSITIVO.")
            self.color_mask[mask] = [0, 1, 0, 0.5]  # Verde para positivo
        else:
            self.update_status(f"Superpixel {clicked_id} desmarcado (NEGATIVO).")
            self.color_mask[mask] = [0, 0, 0, 0]    # Transparente para negativo

        # A CORREÇÃO: Atualiza os dados do artista da máscara em vez de redesenhar
        if self.mask_artist:
            self.mask_artist.set_data(self.color_mask)
        self.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        
        # Fator de zoom
        scale_factor = 1.1 if event.step > 0 else 1 / 1.1
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Posição do mouse
        x, y = event.xdata, event.ydata

        # Calcula os novos limites
        new_xlim = [
            x - (x - xlim[0]) / scale_factor,
            x + (xlim[1] - x) / scale_factor
        ]
        new_ylim = [
            y - (y - ylim[0]) / scale_factor,
            y + (ylim[1] - y) / scale_factor
        ]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def on_motion(self, event):
        if not self._is_panning or event.inaxes != self.ax:
            return

        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        
        # Inverte os valores para o movimento ser natural
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        
        self.canvas.draw_idle()

        # Atualiza a posição inicial para o próximo movimento
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def reset_view(self):
        if self.original_image is not None:
            h, w, _ = self.original_image.shape
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0) # Invertido para imshow
            self.canvas.draw_idle()

    def update_visualization(self):
        if self.original_image is None:
            return
            
        self.ax.clear()
        self.ax.imshow(mark_boundaries(self.original_image, self.superpixels, color=(0, 0, 0)))
        # Cria o artista da máscara na primeira vez que a imagem é desenhada
        self.mask_artist = self.ax.imshow(self.color_mask)
        self.ax.set_title("Clique Esquerdo: Rotular/Desrotular | Botão Meio: Resetar Zoom")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.reset_view() # Garante que a visualização comece sem zoom
        self.canvas.draw()

    def save_results(self):
        if self.labels_df is None or self.image_path is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para salvar.")
            return

        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Salvar CSV apenas com rótulos
        csv_path = filedialog.asksaveasfilename(
            title="Salvar CSV com Rótulos",
            initialfile=f"rotulos_{base_name}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if csv_path:
            try:
                # Salva o DataFrame completo com IDs e rótulos (0 ou 1)
                self.labels_df.to_csv(csv_path, index=False)
                self.update_status(f"CSV de rótulos salvo em {os.path.basename(csv_path)}")
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo CSV.\n\n{e}")
                return

        # Salvar Imagem de visualização
        image_save_path = filedialog.asksaveasfilename(
            title="Salvar imagem com marcações",
            initialfile=f"visualizacao_{base_name}.png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        if image_save_path:
            try:
                self.fig.savefig(image_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
                self.update_status("CSV e Imagem de visualização salvos com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar a imagem.\n\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SuperpixelLabelerLiteGUI(root)
    root.geometry("800x800")
    root.mainloop()

