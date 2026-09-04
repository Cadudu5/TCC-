"""Rotulador interativo e multiplataforma de superpixels.

O rotulador salva somente ``superpixel_id,label``. Os atributos científicos
são extraídos depois, pelo pipeline compartilhado em ``features.extract``.
Assim, abrir uma imagem exige apenas a segmentação SLIC e permanece rápido.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd


def _configure_frozen_matplotlib_cache() -> None:
    """Mantém o cache de fontes entre execuções do pacote PyInstaller."""
    if not getattr(sys, "frozen", False):
        return
    if sys.platform == "win32":
        cache_root = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
    elif sys.platform == "darwin":
        cache_root = Path.home() / "Library" / "Caches"
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_path = cache_root / "RotuladorNeutrofilos" / "matplotlib"
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["MPLCONFIGDIR"] = str(cache_path)


_configure_frozen_matplotlib_cache()

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage.segmentation import mark_boundaries

from features.extract import (
    SLIC_COMPACTNESS,
    SLIC_N_SEGMENTS,
    SLIC_SIGMA,
    segment_superpixels,
)


IMAGE_FILETYPES = [
    ("Imagens", "*.tif *.tiff *.png *.jpg *.jpeg"),
    ("Todos os arquivos", "*.*"),
]


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Carrega a primeira página da imagem como RGB uint8."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("A imagem não pôde ser convertida para RGB")
    return rgb


def new_labels_table(superpixels: np.ndarray) -> pd.DataFrame:
    ids = np.unique(superpixels).astype(np.int32)
    return pd.DataFrame(
        {"superpixel_id": ids, "label": np.zeros(ids.size, dtype=np.int8)}
    )


def read_labels_table(path: str | Path, expected_ids: np.ndarray) -> pd.DataFrame:
    """Lê uma marcação e garante que ela pertence à segmentação carregada."""
    try:
        table = pd.read_csv(path, usecols=["superpixel_id", "label"])
    except ValueError as error:
        raise ValueError(
            "O CSV deve conter exatamente as colunas superpixel_id e label."
        ) from error

    if table[["superpixel_id", "label"]].isna().any().any():
        raise ValueError("O CSV contém valores vazios.")
    if table["superpixel_id"].duplicated().any():
        raise ValueError("O CSV contém IDs de superpixel duplicados.")

    try:
        numeric_ids = pd.to_numeric(table["superpixel_id"], errors="raise")
        numeric_labels = pd.to_numeric(table["label"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("IDs e rótulos precisam ser números inteiros.") from error
    if not np.equal(numeric_ids, np.floor(numeric_ids)).all() or not np.equal(
        numeric_labels, np.floor(numeric_labels)
    ).all():
        raise ValueError("IDs e rótulos precisam ser números inteiros.")
    table = pd.DataFrame(
        {
            "superpixel_id": numeric_ids.astype(np.int32),
            "label": numeric_labels.astype(np.int8),
        }
    )

    if not table["label"].isin([0, 1]).all():
        raise ValueError("Os rótulos válidos são somente 0 e 1.")

    table = table.sort_values("superpixel_id").reset_index(drop=True)
    actual_ids = table["superpixel_id"].to_numpy(dtype=np.int32)
    expected = np.sort(np.asarray(expected_ids, dtype=np.int32))
    if not np.array_equal(actual_ids, expected):
        missing = np.setdiff1d(expected, actual_ids).size
        extra = np.setdiff1d(actual_ids, expected).size
        raise ValueError(
            "A marcação não corresponde à segmentação desta imagem "
            f"({missing} IDs ausentes e {extra} IDs inesperados)."
        )
    return table


def write_labels_atomic(table: pd.DataFrame, path: str | Path) -> None:
    """Salva o contrato mínimo do rótulo sem deixar CSV parcial."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    table.loc[:, ["superpixel_id", "label"]].to_csv(temporary, index=False)
    os.replace(temporary, destination)


class SuperpixelLabelerLiteGUI:
    def __init__(self, root: tk.Tk, initial_image: str | None = None):
        self.root = root
        self.root.title("Rotulador de neutrófilos por superpixels")
        self.root.geometry("1000x850")
        self.root.minsize(760, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.image_path: Path | None = None
        self.original_image: np.ndarray | None = None
        self.superpixels: np.ndarray | None = None
        self.labels_df: pd.DataFrame | None = None
        self.color_mask: np.ndarray | None = None
        self.id_to_row: dict[int, int] = {}
        self.history: list[tuple[int, int]] = []
        self.dirty = False

        self._is_panning = False
        self._pan_start_x: float | None = None
        self._pan_start_y: float | None = None
        self.mask_artist = None

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        toolbar = tk.Frame(main_frame)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        tk.Button(toolbar, text="Carregar imagem", command=self.load_image).pack(
            side=tk.LEFT
        )
        self.btn_open_labels = tk.Button(
            toolbar,
            text="Abrir marcação",
            command=self.load_existing_labels,
            state=tk.DISABLED,
        )
        self.btn_open_labels.pack(side=tk.LEFT, padx=(6, 0))
        self.btn_undo = tk.Button(
            toolbar, text="Desfazer", command=self.undo, state=tk.DISABLED
        )
        self.btn_undo.pack(side=tk.LEFT, padx=(6, 0))
        self.btn_save = tk.Button(
            toolbar,
            text="Salvar marcação",
            command=self.save_results,
            state=tk.DISABLED,
        )
        self.btn_save.pack(side=tk.LEFT, padx=(6, 0))

        self.status = tk.StringVar(value="Carregue uma imagem para começar.")
        tk.Label(
            main_frame,
            textvariable=self.status,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
        ).pack(side=tk.BOTTOM, fill=tk.X)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10
        )

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.root.bind_all("<Control-s>", lambda _event: self.save_results())
        self.root.bind_all("<Command-s>", lambda _event: self.save_results())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Command-z>", lambda _event: self.undo())

        if initial_image:
            self.root.after(100, lambda: self.load_image_path(initial_image))

    def update_status(self, message: str) -> None:
        self.status.set(message)
        self.root.update_idletasks()

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        return messagebox.askyesno(
            "Alterações não salvas",
            "Há marcações não salvas. Deseja descartá-las?",
        )

    def load_image(self) -> None:
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            title="Selecione uma imagem", filetypes=IMAGE_FILETYPES
        )
        if path:
            self.load_image_path(path)

    def load_image_path(self, path: str | Path) -> None:
        self.update_status(f"Carregando {Path(path).name}...")
        try:
            image = load_rgb_image(path)
            self.update_status("Calculando superpixels SLIC...")
            superpixels = segment_superpixels(image)
            table = new_labels_table(superpixels)

            self.image_path = Path(path)
            self.original_image = image
            self.superpixels = superpixels
            self.labels_df = table
            self.id_to_row = {
                int(superpixel_id): row
                for row, superpixel_id in enumerate(table["superpixel_id"])
            }
            self.color_mask = np.zeros((*image.shape[:2], 4), dtype=np.float32)
            self.history.clear()
            self.dirty = False
            self.btn_open_labels.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)
            self.btn_undo.config(state=tk.DISABLED)
            self.update_visualization()
            self._update_ready_status()
        except Exception as error:
            messagebox.showerror(
                "Erro", f"Não foi possível abrir ou segmentar a imagem.\n\n{error}"
            )
            self.update_status("Erro ao carregar a imagem.")

    def load_existing_labels(self) -> None:
        if self.superpixels is None or not self._confirm_discard():
            return
        initial = ""
        if self.image_path is not None:
            initial = f"rotulos_{self.image_path.stem}.csv"
        path = filedialog.askopenfilename(
            title="Abrir CSV de marcação",
            initialfile=initial,
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if not path:
            return
        try:
            table = read_labels_table(path, np.unique(self.superpixels))
            self.labels_df = table
            self.id_to_row = {
                int(superpixel_id): row
                for row, superpixel_id in enumerate(table["superpixel_id"])
            }
            self.history.clear()
            self.btn_undo.config(state=tk.DISABLED)
            self.dirty = False
            self._rebuild_mask()
            self._update_ready_status(prefix=f"Marcação {Path(path).name} carregada.")
        except Exception as error:
            messagebox.showerror("Marcação incompatível", str(error))

    def _rebuild_mask(self) -> None:
        if self.labels_df is None or self.superpixels is None:
            return
        self.color_mask = np.zeros((*self.superpixels.shape, 4), dtype=np.float32)
        positive_ids = self.labels_df.loc[
            self.labels_df["label"] == 1, "superpixel_id"
        ].to_numpy()
        if positive_ids.size:
            selected = np.isin(self.superpixels, positive_ids)
            self.color_mask[selected] = (0.0, 1.0, 0.0, 0.48)
        if self.mask_artist is not None:
            self.mask_artist.set_data(self.color_mask)
            self.canvas.draw_idle()

    def _positive_count(self) -> int:
        if self.labels_df is None:
            return 0
        return int(self.labels_df["label"].sum())

    def _update_ready_status(self, prefix: str = "Pronto.") -> None:
        self.update_status(
            f"{prefix} Positivos: {self._positive_count()} | "
            "esquerdo=marcar, scroll=zoom, direito+arrastar=mover, meio=resetar"
        )

    def _set_label(self, superpixel_id: int, value: int) -> None:
        if self.labels_df is None or self.superpixels is None or self.color_mask is None:
            return
        row = self.id_to_row[superpixel_id]
        self.labels_df.at[row, "label"] = value
        mask = self.superpixels == superpixel_id
        self.color_mask[mask] = (
            (0.0, 1.0, 0.0, 0.48) if value else (0.0, 0.0, 0.0, 0.0)
        )
        if self.mask_artist is not None:
            self.mask_artist.set_data(self.color_mask)
        self.dirty = True

    def on_mouse_press(self, event) -> None:
        if event.inaxes != self.ax or self.superpixels is None:
            return
        if event.button == 1:
            self.on_click_label(event)
        elif event.button == 2:
            self.reset_view()
        elif event.button == 3:
            self._is_panning = True
            self._pan_start_x = event.x
            self._pan_start_y = event.y

    def on_mouse_release(self, event) -> None:
        if event.button == 3:
            self._is_panning = False
            self._pan_start_x = None
            self._pan_start_y = None

    def on_click_label(self, event) -> None:
        if self.original_image is None or self.superpixels is None:
            return
        try:
            x, y = int(event.xdata), int(event.ydata)
        except (TypeError, ValueError):
            return
        height, width = self.superpixels.shape
        if not (0 <= x < width and 0 <= y < height):
            return
        superpixel_id = int(self.superpixels[y, x])
        row = self.id_to_row[superpixel_id]
        previous = int(self.labels_df.at[row, "label"])
        self.history.append((superpixel_id, previous))
        self._set_label(superpixel_id, 1 - previous)
        self.btn_undo.config(state=tk.NORMAL)
        self._update_ready_status()
        self.canvas.draw_idle()

    def undo(self) -> None:
        if not self.history:
            return
        superpixel_id, previous = self.history.pop()
        self._set_label(superpixel_id, previous)
        if not self.history:
            self.btn_undo.config(state=tk.DISABLED)
        self._update_ready_status(prefix=f"Ação no superpixel {superpixel_id} desfeita.")
        self.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        factor = 1.2 if event.step > 0 else 1 / 1.2
        x, y = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(
            x - (x - xlim[0]) / factor,
            x + (xlim[1] - x) / factor,
        )
        self.ax.set_ylim(
            y - (y - ylim[0]) / factor,
            y + (ylim[1] - y) / factor,
        )
        self.canvas.draw_idle()

    def on_motion(self, event) -> None:
        if (
            not self._is_panning
            or event.inaxes != self.ax
            or self._pan_start_x is None
            or self._pan_start_y is None
        ):
            return
        width = max(float(self.ax.bbox.width), 1.0)
        height = max(float(self.ax.bbox.height), 1.0)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        dx = (event.x - self._pan_start_x) * (xlim[1] - xlim[0]) / width
        dy = (event.y - self._pan_start_y) * (ylim[1] - ylim[0]) / height
        self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        if self.superpixels is None:
            return
        height, width = self.superpixels.shape
        self.ax.set_xlim(-0.5, width - 0.5)
        self.ax.set_ylim(height - 0.5, -0.5)
        self.canvas.draw_idle()

    def update_visualization(self) -> None:
        if (
            self.original_image is None
            or self.superpixels is None
            or self.color_mask is None
        ):
            return
        self.ax.clear()
        boundary_image = mark_boundaries(
            self.original_image, self.superpixels, color=(0, 0, 0)
        )
        self.ax.imshow(boundary_image)
        self.mask_artist = self.ax.imshow(self.color_mask)
        self.ax.set_title(self.image_path.name if self.image_path else "Imagem")
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.reset_view()
        self.canvas.draw()

    def save_results(self) -> bool:
        if self.labels_df is None or self.image_path is None:
            return False
        path = filedialog.asksaveasfilename(
            title="Salvar CSV de marcação",
            initialfile=f"rotulos_{self.image_path.stem}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return False
        try:
            csv_path = Path(path)
            write_labels_atomic(self.labels_df, csv_path)
            preview_stem = csv_path.stem.removeprefix("rotulos_")
            preview_path = csv_path.with_name(f"visualizacao_{preview_stem}.png")
            self.fig.savefig(preview_path, dpi=200, bbox_inches="tight", pad_inches=0)
            self.dirty = False
            self._update_ready_status(
                prefix=f"Salvos {csv_path.name} e {preview_path.name}."
            )
            return True
        except Exception as error:
            messagebox.showerror("Erro ao salvar", str(error))
            return False

    def on_close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel(
                "Alterações não salvas", "Deseja salvar antes de fechar?"
            )
            if answer is None:
                return
            if answer and not self.save_results():
                return
        self.root.destroy()


def run_check(image_path: str | Path) -> int:
    image = load_rgb_image(image_path)
    superpixels = segment_superpixels(image)
    print(
        f"OK: {Path(image_path).name} | {image.shape[1]}x{image.shape[0]} | "
        f"{np.unique(superpixels).size} superpixels | "
        f"SLIC(n_segments={SLIC_N_SEGMENTS}, compactness={SLIC_COMPACTNESS:g}, "
        f"sigma={SLIC_SIGMA:g})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="imagem a abrir automaticamente")
    parser.add_argument(
        "--check",
        metavar="IMAGE",
        help="testa o carregamento e o SLIC sem abrir a interface",
    )
    args = parser.parse_args(argv)
    if args.check:
        try:
            return run_check(args.check)
        except Exception as error:
            print(f"ERRO: {error}", file=sys.stderr)
            return 1

    root = tk.Tk()
    SuperpixelLabelerLiteGUI(root, initial_image=args.image)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
