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
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _PROJECT_ROOT = sys._MEIPASS
    _CURRENT_DIR = os.path.join(_PROJECT_ROOT, 'apps')
else:
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from features.extract import segment_superpixels, extract_features


def _artifacts_dir() -> str:
    return os.path.join(_PROJECT_ROOT, 'models', 'artifacts')


def _resolve_model_artifact(base_name: str, meta_name: str) -> tuple[str, str]:
    artifacts_dir = _artifacts_dir()
    booster_path = os.path.join(artifacts_dir, f'{base_name}_booster.json')
    joblib_path = os.path.join(artifacts_dir, f'{base_name}.joblib')
    meta_path = os.path.join(artifacts_dir, meta_name)

    if os.path.exists(booster_path) and os.path.exists(meta_path):
        return booster_path, meta_path
    if os.path.exists(joblib_path) and os.path.exists(meta_path):
        return joblib_path, meta_path
    return booster_path, meta_path


def _load_predictor(model_path: str):
    if model_path.lower().endswith('.json'):
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster
    return joblib.load(model_path)


def _predict_binary(predictor, X_df):
    if type(predictor).__module__.startswith('xgboost.core') and type(predictor).__name__ == 'Booster':
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
        scores = predictor.predict(dmatrix)
        return (np.asarray(scores) >= 0.5).astype(np.int32)
    prediction_input = X_df
    if not hasattr(predictor, 'feature_names_in_') and hasattr(X_df, 'to_numpy'):
        prediction_input = X_df.to_numpy()
    return np.asarray(predictor.predict(prediction_input), dtype=np.int32)


def compute_area_statistics(labels, background_ids, positive_ids):
    """Retorna área positiva, área de tecido e percentual, todos por pixel."""
    labels_array = np.asarray(labels)
    background_array = np.fromiter(background_ids or (), dtype=np.int32)
    positive_array = np.fromiter(positive_ids or (), dtype=np.int32)
    background_mask = np.isin(labels_array, background_array)
    tissue_mask = ~background_mask
    positive_mask = np.isin(labels_array, positive_array) & tissue_mask
    positive_pixels = int(np.count_nonzero(positive_mask))
    tissue_pixels = int(np.count_nonzero(tissue_mask))
    percentage = (100.0 * positive_pixels / tissue_pixels) if tissue_pixels else 0.0
    return positive_pixels, tissue_pixels, percentage


class InferenciaGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Inferência Neutrófilos - Overlay")

        # Estado
        self.image_path = None
        self.image_rgb = None
        self.overlay_image = None
        self._current_pil_image = None
        self._display_width = 1
        self._display_height = 1
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
        self._feature_workers = self._resolve_feature_workers()

        # O artigo selecionou Random Forest para a classificação de neutrófilos.
        self.model_name = 'Random Forest'
        self.model_paths = self._load_primary_model()

        # UI: seleção de imagem, parâmetros e ações
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.btn_load = tk.Button(btn_frame, text="Carregar imagem", command=self.on_load_image)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        # Valor fixo de n_segments para segmentação SLIC
        self.n_segments_var = tk.IntVar(value=5000)

        # Botão marcar fundo
        self.btn_remove_bg = tk.Button(btn_frame, text="Marcar fundo", command=self.on_remove_background, state=tk.DISABLED)
        self.btn_remove_bg.pack(side=tk.LEFT, padx=8)

        # Botão analisar
        self.btn_analyze = tk.Button(btn_frame, text="Analisar imagem", command=self.on_analyze, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=8)
        self._update_analyze_enabled()

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
        self.canvas = tk.Canvas(self.view_frame, bg='#333333', bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind('<Configure>', self._on_resize)

        # UI da aba Estatísticas
        stats_inner = tk.Frame(self.stats_frame)
        stats_inner.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        self.stats_pos_var = tk.StringVar(value="0")
        self.stats_neg_var = tk.StringVar(value="0")
        self.stats_bg_var = tk.StringVar(value="0")
        self.stats_pos_pixels_var = tk.StringVar(value="0")
        self.stats_tissue_pixels_var = tk.StringVar(value="0")
        self.stats_pct_var = tk.StringVar(value="0.00%")
        self.stats_note_var = tk.StringVar(value="")

        tk.Label(stats_inner, text="Positivos (neutrófilo):").grid(row=0, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_pos_var).grid(row=0, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="Negativos (não neutrófilo):").grid(row=1, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_neg_var).grid(row=1, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="Fundo:").grid(row=2, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_bg_var).grid(row=2, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="Área positiva (pixels):").grid(row=3, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_pos_pixels_var).grid(row=3, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="Área tecidual (pixels):").grid(row=4, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_tissue_pixels_var).grid(row=4, column=1, sticky='w', pady=2)
        tk.Label(stats_inner, text="% da área positiva (excluindo fundo):").grid(row=5, column=0, sticky='w', padx=(0,8), pady=2)
        tk.Label(stats_inner, textvariable=self.stats_pct_var).grid(row=5, column=1, sticky='w', pady=2)

        # Observações e ação
        tk.Label(self.stats_frame, textvariable=self.stats_note_var, fg="gray").pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,8))
        self.btn_save_stats = tk.Button(self.stats_frame, text="Salvar CSV", command=self.on_save_stats_csv)
        self.btn_save_stats.pack(side=tk.TOP, padx=8, pady=(0,8), anchor='w')

    def _resolve_feature_workers(self) -> int:
        raw_value = os.environ.get("TCC_INFERENCE_WORKERS", "").strip()
        if raw_value:
            try:
                return max(1, int(raw_value))
            except ValueError:
                return 1

        cpu_count = os.cpu_count() or 1
        if cpu_count <= 2:
            return 1
        return min(cpu_count - 1, 8)

    def _load_primary_model(self) -> tuple[str, str] | None:
        artifacts_dir = _artifacts_dir()
        model_path, meta_path = _resolve_model_artifact('rf_best_cv', 'rf_best_cv_meta.json')
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            messagebox.showerror(
                "Erro",
                f"Modelo Random Forest não encontrado em\n{artifacts_dir}\nTreine e salve o modelo antes de usar a ferramenta.")
            return None
        return model_path, meta_path

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
        self._current_pil_image = Image.fromarray(self.image_rgb)
        self._update_image_display()
        self.btn_save.configure(state=tk.DISABLED)

    def _on_resize(self, event):
        self._display_width = event.width
        self._display_height = event.height
        self._update_image_display()

    def _update_image_display(self):
        if self._current_pil_image is None:
            return
        
        if self._display_width < 1 or self._display_height < 1:
            return

        img_w, img_h = self._current_pil_image.size
        
        # Calculate scale to fit within display area while maintaining aspect ratio
        scale = min(self._display_width / img_w, self._display_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        if new_w < 1 or new_h < 1:
            return

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS

        resized = self._current_pil_image.resize((new_w, new_h), resample)
        self.tk_img = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        # Center image
        x = self._display_width // 2
        y = self._display_height // 2
        self.canvas.create_image(x, y, image=self.tk_img, anchor='center')

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
        df = extract_features(image_rgb, labels, max_workers=self._feature_workers)
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
                self.btn_remove_bg.configure(state=tk.DISABLED)
                # Nada para estatísticas aqui (aba permanece)
            else:
                self.progress.stop()
                self.btn_load.configure(state=tk.NORMAL)
                # reavalia se podemos habilitar remoção e análise
                self._update_analyze_enabled()
                # Nada para estatísticas aqui (aba permanece)
        self.root.after(0, _apply)

    def _update_analyze_enabled(self):
        has_image = (self.image_rgb is not None)
        key = self._get_cache_key(int(self.n_segments_var.get())) if has_image else None
        has_current_background = self._bg_sp_ids is not None and self._bg_cache_key == key
        enabled_analyze = has_image and self.model_paths is not None and has_current_background
        self.btn_analyze.configure(state=(tk.NORMAL if enabled_analyze else tk.DISABLED))
        # Remover fundo depende apenas de haver imagem carregada
        self.btn_remove_bg.configure(state=(tk.NORMAL if has_image else tk.DISABLED))
        # Estatísticas disponível se há imagem
        # Aba Estatísticas sempre visível; apenas valores refletem estado

    def _get_bg_model_paths(self):
        candidates = (
            ('fundo_xgb_artigo', 'fundo_xgb_artigo_meta.json'),
            ('fundo_xgb', 'fundo_xgb_meta.json'),
        )
        for base_name, meta_name in candidates:
            model_path, meta_path = _resolve_model_artifact(base_name, meta_name)
            if os.path.exists(model_path) and os.path.exists(meta_path):
                return model_path, meta_path
        raise FileNotFoundError(
            "Modelo XGBoost de fundo não encontrado. Execute:\n"
            "python models/treinamento_fundo_artigo.py"
        )

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
            bg_model = _load_predictor(model_path)
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
            y_bg = _predict_binary(bg_model, df[feature_names])

            # Construir máscara de fundo
            sp_ids = df['superpixel_id'].to_numpy(dtype=np.int32)
            bg_ids_array = sp_ids[np.asarray(y_bg, dtype=np.int32) == 1]
            mask_bg = np.isin(labels, bg_ids_array)
            bg_ids = set(map(int, bg_ids_array.tolist()))

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
                self._pred_cache_key = None
                self._pred_bg_cache_key = None
                self._pred_pos_sp_ids = None
                self._pred_neg_sp_ids = None
                self.overlay_image = overlay
                # Atualiza estatísticas (fundo pode ter mudado)
                self._update_stats_view(n_segments=n_segments)
                
                self._current_pil_image = Image.fromarray(self.overlay_image)
                self._update_image_display()
                
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
        if self.model_paths is None:
            messagebox.showwarning("Aviso", "Modelo Random Forest não disponível para análise.")
            return
        try:
            n_segments = int(self.n_segments_var.get())
        except Exception:
            messagebox.showerror("Erro", "Valor de n_segments inválido.")
            return
        if self._bg_sp_ids is None or self._bg_cache_key != self._get_cache_key(n_segments):
            messagebox.showwarning("Aviso", "Execute 'Marcar fundo' antes de analisar a imagem.")
            return

        self._set_status("Iniciando análise...")
        self._set_busy(True)

        thread = threading.Thread(
            target=self._analyze_worker,
            args=(self.image_rgb.copy(), n_segments),
            daemon=True,
        )
        thread.start()

    def _analyze_worker(self, image_rgb: np.ndarray, n_segments: int):
        try:
            self._set_status("Segmentando (SLIC)...")
            labels, df = self._get_labels_and_features(image_rgb, n_segments)

            self._set_status("Carregando modelo...")
            if self.model_paths is None:
                raise RuntimeError("Modelo Random Forest não está disponível.")
            model_path, meta_path = self.model_paths
            model = _load_predictor(model_path)
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
                    
                    self._current_pil_image = Image.fromarray(self.overlay_image)
                    self._update_image_display()
                    
                    self.btn_save.configure(state=tk.NORMAL)
                    self._set_status("Sem superpixels para análise (após remover fundo)")
                    self._update_stats_view(n_segments=n_segments)
                    self._set_busy(False)
                self.root.after(0, _finish_empty)
                return

            self._set_status("Predizendo superpixels...")
            y_pred = _predict_binary(model, valid_df[feature_names])

            self._set_status("Gerando overlay...")
            # Começa do overlay atual (que pode conter o fundo em vermelho) ou da imagem original
            base = self.overlay_image if (self.background_mask is not None and self._bg_cache_key == self._get_cache_key(n_segments)) else image_rgb
            overlay = base.copy().astype(np.float32)
            alpha = 100  # 0-255

            valid_sp_ids = valid_sp_ids.astype(np.int32, copy=False)
            pos_ids_array = valid_sp_ids[np.asarray(y_pred, dtype=np.int32) == 1]
            mask_pos = np.isin(labels, pos_ids_array)

            green = np.zeros_like(overlay)
            green[..., 1] = 255
            a = (alpha / 255.0).astype(np.float32) if isinstance(alpha, np.ndarray) else float(alpha) / 255.0
            overlay[mask_pos] = (1.0 - a) * overlay[mask_pos] + a * green[mask_pos]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            def _finish():
                # Guarda predições para estatísticas
                pos_ids = set(map(int, pos_ids_array.tolist()))
                neg_ids = set(map(int, valid_sp_ids)) - pos_ids
                self._pred_pos_sp_ids = pos_ids
                self._pred_neg_sp_ids = neg_ids
                self._pred_cache_key = self._get_cache_key(n_segments)
                self._pred_bg_cache_key = self._bg_cache_key
                self.overlay_image = overlay
                
                self._current_pil_image = Image.fromarray(self.overlay_image)
                self._update_image_display()
                
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
            return 0, 0, 0, 0, 0, 0.0, ""
        key = self._get_cache_key(n_segments)
        bg_count = 0
        if self._bg_sp_ids is not None and self._bg_cache_key == key:
            bg_count = len(self._bg_sp_ids)
        if not (self._pred_cache_key == key and self._pred_pos_sp_ids is not None and self._pred_neg_sp_ids is not None):
            return 0, 0, bg_count, 0, 0, 0.0, ""
        note = ""
        if self._pred_bg_cache_key != self._bg_cache_key:
            note = "Fundo alterado após a análise. Reanalise para atualizar as estatísticas."
        pos_count = len(self._pred_pos_sp_ids)
        neg_count = len(self._pred_neg_sp_ids)
        labels, _ = self._cache[key]
        background_ids = self._bg_sp_ids if self._bg_cache_key == key else set()
        positive_pixels, tissue_pixels, pct_pos = compute_area_statistics(
            labels, background_ids, self._pred_pos_sp_ids
        )
        return pos_count, neg_count, bg_count, positive_pixels, tissue_pixels, pct_pos, note

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
            self.stats_pos_pixels_var.set("0")
            self.stats_tissue_pixels_var.set("0")
            self.stats_pct_var.set("0.00%")
            self.stats_note_var.set("")
            return
        pos, neg, bg, pos_pixels, tissue_pixels, pct, note = self._compute_stats(n_segments)
        self.stats_pos_var.set(str(pos))
        self.stats_neg_var.set(str(neg))
        self.stats_bg_var.set(str(bg))
        self.stats_pos_pixels_var.set(str(pos_pixels))
        self.stats_tissue_pixels_var.set(str(tissue_pixels))
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
        pos, neg, bg, pos_pixels, tissue_pixels, pct, note = self._compute_stats(n_segments)
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
                writer.writerow([
                    "superpixels_positivos_neutrofilo",
                    "superpixels_negativos_nao_neutrofilo",
                    "superpixels_fundo",
                    "area_positiva_pixels",
                    "area_tecidual_pixels",
                    "percentual_area_positiva_excluindo_fundo",
                    "observacao",
                ])
                writer.writerow([pos, neg, bg, pos_pixels, tissue_pixels, f"{pct:.2f}", note])
        except Exception as e:
            messagebox.showerror("Erro ao salvar CSV", str(e))
            return
        messagebox.showinfo("Sucesso", f"Estatísticas salvas em:\n{out_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = InferenciaGUI(root)
    root.mainloop()
