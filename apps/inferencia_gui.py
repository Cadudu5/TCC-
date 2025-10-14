import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib

# Garante que o diretório raiz do projeto esteja no sys.path para importar 'features'
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from features.extract import segment_superpixels, extract_features


class InferenciaGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Inferência Neutrófilos - Overlay")

        # Estado
        self.image_path = None
        self.image_rgb = None
        self.overlay_image = None

        # Carregar artefatos do modelo
        self._load_artifacts()

        # UI simples: dois botões e um canvas para preview
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.btn_load = tk.Button(btn_frame, text="Carregar imagem", command=self.on_load_image)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        self.btn_save = tk.Button(btn_frame, text="Salvar overlay", state=tk.DISABLED, command=self.on_save_overlay)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Label(root)
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)

    def _load_artifacts(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(os.path.dirname(base_dir), 'models', 'artifacts')
        model_path = os.path.join(artifacts_dir, 'rf_best.joblib')
        meta_path = os.path.join(artifacts_dir, 'rf_best_meta.json')

        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            messagebox.showerror("Erro", f"Artefatos não encontrados em\n{artifacts_dir}\nTreine o modelo primeiro.")
            raise SystemExit(1)

        self.model = joblib.load(model_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

    def on_load_image(self):
        path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[
                ("Imagens", ".tif .tiff .png .jpg .jpeg"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not path:
            return

        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            messagebox.showerror("Erro ao abrir imagem", str(e))
            return

        self.image_path = path
        self.image_rgb = np.array(img)

        try:
            self.overlay_image = self._run_inference(self.image_rgb)
        except Exception as e:
            messagebox.showerror("Erro na inferência", str(e))
            return

        # Mostrar preview
        preview = Image.fromarray(self.overlay_image)
        preview_tk = ImageTk.PhotoImage(preview)
        self.canvas.configure(image=preview_tk)
        self.canvas.image = preview_tk
        self.btn_save.configure(state=tk.NORMAL)

    def _run_inference(self, image_rgb: np.ndarray) -> np.ndarray:
        slic_params = self.meta.get('slic', {"n_segments": 5000, "compactness": 10, "sigma": 3})
        labels = segment_superpixels(
            image_rgb,
            n_segments=slic_params.get('n_segments', 5000),
            compactness=slic_params.get('compactness', 10),
            sigma=slic_params.get('sigma', 3),
        )

        df = extract_features(image_rgb, labels)
        feature_names = self.meta['feature_names']
        # Garante colunas na ordem correta; se faltarem, lança erro mais claro
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise RuntimeError(f"Features ausentes na extração: {missing}")

        y_pred = self.model.predict(df[feature_names])

        # Cria overlay verde onde a predição == 1
        overlay = image_rgb.copy()
        alpha = 100  # 0-255

        # Máscara por superpixel
        mask_pos = np.zeros(labels.shape, dtype=bool)
        # Mapear superpixel_id -> pred
        sp_ids = df['superpixel_id'].to_numpy()
        for sp_id, pred in zip(sp_ids, y_pred):
            if int(pred) == 1:
                mask_pos[labels == int(sp_id)] = True

        # Aplicar verde com alpha
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        # alpha blend: out = (1-a)*orig + a*green
        a = (alpha / 255.0).astype(np.float32) if isinstance(alpha, np.ndarray) else float(alpha) / 255.0
        overlay = overlay.astype(np.float32)
        overlay[mask_pos] = (1.0 - a) * overlay[mask_pos] + a * green[mask_pos]
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay

    def on_save_overlay(self):
        if self.overlay_image is None or self.image_path is None:
            return
        initial_dir = os.path.dirname(self.image_path)
        initial_file = 'overlay.png'
        out_path = filedialog.asksaveasfilename(
            title="Salvar overlay",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("Imagem", "*.png;*.jpg;*.jpeg"),
                ("Todos os arquivos", "*.*"),
            ],
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        if not out_path:
            return
        try:
            Image.fromarray(self.overlay_image).save(out_path)
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))
            return
        messagebox.showinfo("Sucesso", f"Overlay salvo em:\n{out_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = InferenciaGUI(root)
    root.mainloop()


