import os
import sys
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
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

        # Cache de resultados por (image_path, n_segments)
        self._cache = {}

        # Descobrir modelos disponíveis
        self.available_models = self._discover_models()

        # UI: seleção de imagem, parâmetros e ações
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.btn_load = tk.Button(btn_frame, text="Carregar imagem", command=self.on_load_image)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        # Parâmetro SLIC: n_segments (compactness=10, sigma=3 fixos)
        tk.Label(btn_frame, text="n_segments:").pack(side=tk.LEFT, padx=(12, 4))
        self.n_segments_var = tk.IntVar(value=5000)
        self.spin_segments = tk.Spinbox(
            btn_frame,
            from_=500,
            to=30000,
            increment=500,
            textvariable=self.n_segments_var,
            width=7
        )
        self.spin_segments.pack(side=tk.LEFT, padx=4)

        # Seletor de modelo
        tk.Label(btn_frame, text="Modelo:").pack(side=tk.LEFT, padx=(12, 4))
        self.model_var = tk.StringVar()
        self.combo_model = ttk.Combobox(btn_frame, textvariable=self.model_var, state='readonly', width=18)
        self.combo_model.pack(side=tk.LEFT, padx=4)

        # Botão analisar
        self.btn_analyze = tk.Button(btn_frame, text="Analisar imagem", command=self.on_analyze, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=8)
        self._populate_model_selector()

        # Botão salvar
        self.btn_save = tk.Button(btn_frame, text="Salvar overlay", state=tk.DISABLED, command=self.on_save_overlay)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        # Barra de progresso e status
        status_frame = tk.Frame(root)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.status_var = tk.StringVar(value="Pronto")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, anchor='w')
        self.status_label.pack(side=tk.LEFT)

        # Canvas de preview
        self.canvas = tk.Label(root)
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)

    def _discover_models(self) -> dict:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(os.path.dirname(base_dir), 'models', 'artifacts')
        candidates = {
            'RandomForest': ('rf_best.joblib', 'rf_best_meta.json'),
            'SVM': ('svm_best.joblib', 'svm_best_meta.json'),
            'XGBoost': ('xgb_best.joblib', 'xgb_best_meta.json'),
            'NeuralNet': ('nn_best.joblib', 'nn_best_meta.json'),
        }
        available = {}
        for name, (mfile, metafile) in candidates.items():
            model_path = os.path.join(artifacts_dir, mfile)
            meta_path = os.path.join(artifacts_dir, metafile)
            if os.path.exists(model_path) and os.path.exists(meta_path):
                available[name] = (model_path, meta_path)
        if not available:
            messagebox.showerror(
                "Erro",
                f"Nenhum modelo encontrado em\n{artifacts_dir}\nTreine e salve ao menos um modelo.")
        return available

    def _populate_model_selector(self):
        names = list(self.available_models.keys())
        self.combo_model['values'] = names
        if names:
            self.combo_model.current(0)
        # Habilita o botão analisar somente se houver modelo e imagem
        self._update_analyze_enabled()

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
        self.overlay_image = None
        # Limpa cache para outras imagens? Mantemos cache por (image_path, n_segments)
        # Apenas reabilita análise
        self._update_analyze_enabled()

        # Mostrar preview da imagem original
        preview = Image.fromarray(self.image_rgb)
        preview_tk = ImageTk.PhotoImage(preview)
        self.canvas.configure(image=preview_tk)
        self.canvas.image = preview_tk
        self.btn_save.configure(state=tk.DISABLED)

    def _get_cache_key(self, n_segments: int) -> tuple:
        return (self.image_path, int(n_segments))

    def _get_labels_and_features(self, image_rgb: np.ndarray, n_segments: int):
        key = self._get_cache_key(n_segments)
        if key in self._cache:
            return self._cache[key]

        labels = segment_superpixels(
            image_rgb,
            n_segments=int(n_segments),
            compactness=10,
            sigma=3,
        )
        df = extract_features(image_rgb, labels)
        self._cache[key] = (labels, df)
        return labels, df

    def _set_status(self, text: str):
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_busy(self, busy: bool):
        def _apply():
            if busy:
                self.progress.start(10)
                self.btn_analyze.configure(state=tk.DISABLED)
                self.btn_load.configure(state=tk.DISABLED)
                self.combo_model.configure(state='disabled')
                self.spin_segments.configure(state='disabled')
            else:
                self.progress.stop()
                self.btn_load.configure(state=tk.NORMAL)
                self.combo_model.configure(state='readonly' if self.available_models else 'disabled')
                self.spin_segments.configure(state=tk.NORMAL)
                self._update_analyze_enabled()
        self.root.after(0, _apply)

    def _update_analyze_enabled(self):
        enabled = (self.image_rgb is not None) and bool(self.available_models)
        self.btn_analyze.configure(state=(tk.NORMAL if enabled else tk.DISABLED))

    def on_analyze(self):
        if self.image_rgb is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        if not self.available_models:
            messagebox.showwarning("Aviso", "Nenhum modelo disponível para análise.")
            return
        try:
            n_segments = int(self.n_segments_var.get())
        except Exception:
            messagebox.showerror("Erro", "Valor de n_segments inválido.")
            return
        model_name = self.model_var.get() or (list(self.available_models.keys())[0])

        self._set_status("Iniciando análise...")
        self._set_busy(True)

        thread = threading.Thread(
            target=self._analyze_worker,
            args=(self.image_rgb.copy(), n_segments, model_name),
            daemon=True,
        )
        thread.start()

    def _analyze_worker(self, image_rgb: np.ndarray, n_segments: int, model_name: str):
        try:
            self._set_status("Segmentando (SLIC)...")
            labels, df = self._get_labels_and_features(image_rgb, n_segments)

            self._set_status("Carregando modelo...")
            model_path, meta_path = self.available_models[model_name]
            model = joblib.load(model_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            feature_names = meta.get('feature_names', [])
            if not feature_names:
                raise RuntimeError("'feature_names' ausentes nos metadados do modelo.")

            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise RuntimeError(f"Features ausentes na extração: {missing}")

            self._set_status("Predizendo superpixels...")
            y_pred = model.predict(df[feature_names])

            self._set_status("Gerando overlay...")
            overlay = image_rgb.copy()
            alpha = 100  # 0-255

            mask_pos = np.zeros(labels.shape, dtype=bool)
            sp_ids = df['superpixel_id'].to_numpy()
            for sp_id, pred in zip(sp_ids, y_pred):
                if int(pred) == 1:
                    mask_pos[labels == int(sp_id)] = True

            green = np.zeros_like(overlay)
            green[..., 1] = 255
            a = (alpha / 255.0).astype(np.float32) if isinstance(alpha, np.ndarray) else float(alpha) / 255.0
            overlay = overlay.astype(np.float32)
            overlay[mask_pos] = (1.0 - a) * overlay[mask_pos] + a * green[mask_pos]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            def _finish():
                self.overlay_image = overlay
                preview = Image.fromarray(self.overlay_image)
                preview_tk = ImageTk.PhotoImage(preview)
                self.canvas.configure(image=preview_tk)
                self.canvas.image = preview_tk
                self.btn_save.configure(state=tk.NORMAL)
                self._set_status("Concluído")
                self._set_busy(False)

            self.root.after(0, _finish)

        except Exception as e:
            def _error():
                self._set_busy(False)
                self._set_status("Erro")
                messagebox.showerror("Erro na análise", str(e))
            self.root.after(0, _error)

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


