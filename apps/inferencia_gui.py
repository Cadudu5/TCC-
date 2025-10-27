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
import csv
from datetime import datetime

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
        self.background_mask = None  # máscara booleana 2D dos superpixels de fundo
        self._bg_cache_key = None    # cache key (image_path, n_segments) usado na remoção de fundo
        self._bg_sp_ids = None       # conjunto de superpixel_ids previstos como fundo
        
        # Predições recentes (neutrófilos) para estatísticas
        self._pred_cache_key = None
        self._pred_bg_cache_key = None
        self._pred_pos_sp_ids = None
        self._pred_neg_sp_ids = None
         
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

        # Botão marcar fundo
        self.btn_remove_bg = tk.Button(btn_frame, text="Marcar fundo", command=self.on_remove_background, state=tk.DISABLED)
        self.btn_remove_bg.pack(side=tk.LEFT, padx=8)

        # Botão analisar
        self.btn_analyze = tk.Button(btn_frame, text="Analisar imagem", command=self.on_analyze, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=8)
        self._populate_model_selector()

        # Botão salvar
        self.btn_save = tk.Button(btn_frame, text="Salvar overlay", state=tk.DISABLED, command=self.on_save_overlay)
        self.btn_save.pack(side=tk.LEFT, padx=4)
        
        # Botão estatísticas
        # (Aba Estatísticas substitui popup; botão não é necessário)
         
        # Barra de progresso e status
        status_frame = tk.Frame(root)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.status_var = tk.StringVar(value="Pronto")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, anchor='w')
        self.status_label.pack(side=tk.LEFT)

        # Área principal com abas
        self.notebook = ttk.Notebook(root)
        self.view_frame = tk.Frame(self.notebook)
        self.stats_frame = tk.Frame(self.notebook)
        self.notebook.add(self.view_frame, text='Visualização')
        self.notebook.add(self.stats_frame, text='Estatísticas')
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Canvas de preview (na aba Visualização)
        self.canvas = tk.Label(self.view_frame)
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)

        # UI da aba Estatísticas
        stats_inner = tk.Frame(self.stats_frame)
        stats_inner.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        self.stats_pos_var = tk.StringVar(value="0")
        self.stats_neg_var = tk.StringVar(value="0")
        self.stats_bg_var = tk.StringVar(value="0")
        self.stats_pct_var = tk.StringVar(value="0.00%")
        self.stats_note_var = tk.StringVar(value="")

        tk.Label(stats_inner, text="Positivos (neutrófilo):").grid(row=0, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_pos_var).grid(row=0, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="Negativos (não neutrófilo):").grid(row=1, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_neg_var).grid(row=1, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="Fundo:").grid(row=2, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_bg_var).grid(row=2, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="% Positivo (excluindo fundo):").grid(row=3, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_pct_var).grid(row=3, column=1, sticky='w', pady=2)

        # Observações e ação
        tk.Label(self.stats_frame, textvariable=self.stats_note_var, fg="gray").pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,8))
        self.btn_save_stats = tk.Button(self.stats_frame, text="Salvar CSV", command=self.on_save_stats_csv)
        self.btn_save_stats.pack(side=tk.TOP, padx=8, pady=(0,8), anchor='w')

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
        self.background_mask = None
        self._bg_cache_key = None
        self._bg_sp_ids = None
        # Limpa cache para outras imagens? Mantemos cache por (image_path, n_segments)
        # Apenas reabilita análise
        self._update_analyze_enabled()
        # Limpa predições anteriores e estatísticas
        self._pred_cache_key = None
        self._pred_bg_cache_key = None
        self._pred_pos_sp_ids = None
        self._pred_neg_sp_ids = None
        self._update_stats_view(n_segments=None)
 
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
                self.btn_remove_bg.configure(state=tk.DISABLED)
                # Nada para estatísticas aqui (aba permanece)
            else:
                self.progress.stop()
                self.btn_load.configure(state=tk.NORMAL)
                self.combo_model.configure(state='readonly' if self.available_models else 'disabled')
                self.spin_segments.configure(state=tk.NORMAL)
                # reavalia se podemos habilitar remoção e análise
                self._update_analyze_enabled()
                # Nada para estatísticas aqui (aba permanece)
        self.root.after(0, _apply)

    def _update_analyze_enabled(self):
        has_image = (self.image_rgb is not None)
        enabled_analyze = has_image and bool(self.available_models)
        self.btn_analyze.configure(state=(tk.NORMAL if enabled_analyze else tk.DISABLED))
        # Remover fundo depende apenas de haver imagem carregada
        self.btn_remove_bg.configure(state=(tk.NORMAL if has_image else tk.DISABLED))
        # Estatísticas disponível se há imagem
        # Aba Estatísticas sempre visível; apenas valores refletem estado

    def _get_bg_model_paths(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(os.path.dirname(base_dir), 'models', 'artifacts')
        model_path = os.path.join(artifacts_dir, 'fundo_rf.joblib')
        meta_path = os.path.join(artifacts_dir, 'fundo_rf_meta.json')
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Modelo de fundo não encontrado. Esperado em:\n{model_path}\n{meta_path}\nTreine com models/treinamento_fundo_rf.py."
            )
        return model_path, meta_path

    def on_remove_background(self):
        if self.image_rgb is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        try:
            n_segments = int(self.n_segments_var.get())
        except Exception:
            messagebox.showerror("Erro", "Valor de n_segments inválido.")
            return

        self._set_status("Marcando fundo...")
        self._set_busy(True)

        thread = threading.Thread(
            target=self._remove_bg_worker,
            args=(self.image_rgb.copy(), n_segments),
            daemon=True,
        )
        thread.start()

    def _remove_bg_worker(self, image_rgb: np.ndarray, n_segments: int):
        try:
            # Segmentação e features
            self._set_status("Segmentando (SLIC) para fundo...")
            labels, df = self._get_labels_and_features(image_rgb, n_segments)

            # Carrega modelo de fundo
            self._set_status("Carregando modelo de fundo...")
            model_path, meta_path = self._get_bg_model_paths()
            bg_model = joblib.load(model_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                bg_meta = json.load(f)
            feature_names = bg_meta.get('feature_names', [])
            if not feature_names:
                raise RuntimeError("'feature_names' ausentes nos metadados do modelo de fundo.")
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise RuntimeError(f"Features ausentes na extração (fundo): {missing}")

            # Predição: 1 -> fundo
            self._set_status("Predizendo fundo...")
            y_bg = bg_model.predict(df[feature_names])

            # Construir máscara de fundo
            mask_bg = np.zeros(labels.shape, dtype=bool)
            sp_ids = df['superpixel_id'].to_numpy()
            bg_ids = set()
            for sp_id, pred in zip(sp_ids, y_bg):
                if int(pred) == 1:
                    mask_bg[labels == int(sp_id)] = True
                    bg_ids.add(int(sp_id))

            # Overlay vermelho no fundo
            overlay = image_rgb.copy().astype(np.float32)
            red = np.zeros_like(overlay)
            red[..., 0] = 255
            a = 120 / 255.0
            overlay[mask_bg] = (1.0 - a) * overlay[mask_bg] + a * red[mask_bg]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            def _finish():
                self.background_mask = mask_bg
                self._bg_cache_key = self._get_cache_key(n_segments)
                self._bg_sp_ids = bg_ids
                self.overlay_image = overlay
                # Atualiza estatísticas (fundo pode ter mudado)
                self._update_stats_view(n_segments=n_segments)
                preview = Image.fromarray(self.overlay_image)
                preview_tk = ImageTk.PhotoImage(preview)
                self.canvas.configure(image=preview_tk)
                self.canvas.image = preview_tk
                self.btn_save.configure(state=tk.NORMAL)
                self._set_status("Fundo marcado (vermelho)")
                self._set_busy(False)

            self.root.after(0, _finish)

        except Exception as e:
            _err_msg_bg = str(e)
            def _error():
                self._set_busy(False)
                self._set_status("Erro")
                messagebox.showerror("Erro ao marcar fundo", _err_msg_bg)
            self.root.after(0, _error)

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

            # Se houver máscara de fundo calculada com o mesmo cache key, excluir esses superpixels
            valid_df = df
            valid_sp_ids = df['superpixel_id'].to_numpy()
            if self._bg_sp_ids is not None and self._bg_cache_key == self._get_cache_key(n_segments):
                mask_keep = ~df['superpixel_id'].astype(int).isin(self._bg_sp_ids)
                valid_df = df.loc[mask_keep].reset_index(drop=True)
                valid_sp_ids = valid_df['superpixel_id'].to_numpy()

            # Se não restarem superpixels válidos, finalize sem overlay verde
            if valid_df.shape[0] == 0:
                base = self.overlay_image if (self.background_mask is not None and self._bg_cache_key == self._get_cache_key(n_segments)) else image_rgb
                overlay = base.copy().astype(np.uint8)
                def _finish_empty():
                    self._pred_pos_sp_ids = set()
                    self._pred_neg_sp_ids = set()
                    self._pred_cache_key = self._get_cache_key(n_segments)
                    self._pred_bg_cache_key = self._bg_cache_key
                    self.overlay_image = overlay
                    preview = Image.fromarray(self.overlay_image)
                    preview_tk = ImageTk.PhotoImage(preview)
                    self.canvas.configure(image=preview_tk)
                    self.canvas.image = preview_tk
                    self.btn_save.configure(state=tk.NORMAL)
                    self._set_status("Sem superpixels para análise (após remover fundo)")
                    self._update_stats_view(n_segments=n_segments)
                    self._set_busy(False)
                self.root.after(0, _finish_empty)
                return

            self._set_status("Predizendo superpixels...")
            y_pred = model.predict(valid_df[feature_names])

            self._set_status("Gerando overlay...")
            # Começa do overlay atual (que pode conter o fundo em vermelho) ou da imagem original
            base = self.overlay_image if (self.background_mask is not None and self._bg_cache_key == self._get_cache_key(n_segments)) else image_rgb
            overlay = base.copy().astype(np.float32)
            alpha = 100  # 0-255

            mask_pos = np.zeros(labels.shape, dtype=bool)
            sp_ids = valid_sp_ids
            for sp_id, pred in zip(sp_ids, y_pred):
                if int(pred) == 1:
                    mask_pos[labels == int(sp_id)] = True

            green = np.zeros_like(overlay)
            green[..., 1] = 255
            a = (alpha / 255.0).astype(np.float32) if isinstance(alpha, np.ndarray) else float(alpha) / 255.0
            overlay[mask_pos] = (1.0 - a) * overlay[mask_pos] + a * green[mask_pos]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            def _finish():
                # Guarda predições para estatísticas
                pos_ids = {int(sp_id) for sp_id, pred in zip(valid_sp_ids, y_pred) if int(pred) == 1}
                neg_ids = set(map(int, valid_sp_ids)) - pos_ids
                self._pred_pos_sp_ids = pos_ids
                self._pred_neg_sp_ids = neg_ids
                self._pred_cache_key = self._get_cache_key(n_segments)
                self._pred_bg_cache_key = self._bg_cache_key
                self.overlay_image = overlay
                preview = Image.fromarray(self.overlay_image)
                preview_tk = ImageTk.PhotoImage(preview)
                self.canvas.configure(image=preview_tk)
                self.canvas.image = preview_tk
                self.btn_save.configure(state=tk.NORMAL)
                # Atualiza estatísticas após predição
                self._set_status("Concluído")
                self._update_stats_view(n_segments=n_segments)
                self._set_busy(False)

            self.root.after(0, _finish)

        except Exception as e:
            _err_msg_an = str(e)
            def _error():
                self._set_busy(False)
                self._set_status("Erro")
                messagebox.showerror("Erro na análise", _err_msg_an)
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

    def _compute_stats(self, n_segments: int | None):
        if self.image_rgb is None or n_segments is None:
            return 0, 0, 0, 0.0, ""
        key = self._get_cache_key(n_segments)
        bg_count = 0
        if self._bg_sp_ids is not None and self._bg_cache_key == key:
            bg_count = len(self._bg_sp_ids)
        if not (self._pred_cache_key == key and self._pred_pos_sp_ids is not None and self._pred_neg_sp_ids is not None):
            return 0, 0, bg_count, 0.0, ""
        note = ""
        if self._pred_bg_cache_key != self._bg_cache_key:
            note = "Fundo alterado após a análise. Reanalise para atualizar as estatísticas."
        pos_count = len(self._pred_pos_sp_ids)
        neg_count = len(self._pred_neg_sp_ids)
        denom = pos_count + neg_count
        pct_pos = (100.0 * pos_count / denom) if denom > 0 else 0.0
        return pos_count, neg_count, bg_count, pct_pos, note

    def _update_stats_view(self, n_segments: int | None):
        try:
            if n_segments is None:
                # tenta ler do UI
                n_segments = int(self.n_segments_var.get())
        except Exception:
            # sem valor válido
            self.stats_pos_var.set("0")
            self.stats_neg_var.set("0")
            self.stats_bg_var.set("0")
            self.stats_pct_var.set("0.00%")
            self.stats_note_var.set("")
            return
        pos, neg, bg, pct, note = self._compute_stats(n_segments)
        self.stats_pos_var.set(str(pos))
        self.stats_neg_var.set(str(neg))
        self.stats_bg_var.set(str(bg))
        self.stats_pct_var.set(f"{pct:.2f}%")
        self.stats_note_var.set(note)

    def on_save_stats_csv(self):
        if self.image_rgb is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        try:
            n_segments = int(self.n_segments_var.get())
        except Exception:
            messagebox.showerror("Erro", "Valor de n_segments inválido.")
            return
        pos, neg, bg, pct, note = self._compute_stats(n_segments)
        initial_dir = os.path.dirname(self.image_path) if self.image_path else os.getcwd()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"estatisticas_{ts}.csv"
        out_path = filedialog.asksaveasfilename(
            title="Salvar estatísticas CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
            initialdir=initial_dir,
            initialfile=default_name,
        )
        if not out_path:
            return
        try:
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["positivos_neutrofilo", "negativos_nao_neutrofilo", "fundo", "percentual_positivo_excluindo_fundo", "observacao"])
                writer.writerow([pos, neg, bg, f"{pct:.2f}", note])
        except Exception as e:
            messagebox.showerror("Erro ao salvar CSV", str(e))
            return
        messagebox.showinfo("Sucesso", f"Estatísticas salvas em:\n{out_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = InferenciaGUI(root)
    root.mainloop()


