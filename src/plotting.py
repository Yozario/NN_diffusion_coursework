"""Единый стиль графиков для отчёта и общие функции рисования.

Все скрипты рисуют через этот модуль, поэтому рисунки в отчёте выглядят одинаково:
одни шрифты, цвета серий, сетка и разрешение. Цвета серий выбраны так, чтобы
различаться и при нарушениях цветового зрения; первые два — для двух сравниваемых
вариантов (например, двух расписаний шума).

Использование::

    from src.plotting import apply_style, save_figure, SERIES_COLORS
    apply_style()
    fig, ax = plt.subplots()
    ...
    save_figure(fig, "results/figures/my_plot.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, окно не нужно (работает и без дисплея)

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from cycler import cycler  # noqa: E402

from src.data import to_unit_range  # noqa: E402
from src.utils import ensure_dir, resolve_path  # noqa: E402

# Цвета серий в фиксированном порядке: первая серия всегда синяя, вторая — оранжевая и т. д.
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "red": "#e34948",
}
SERIES_COLORS = list(COLORS.values())

INK = "#0b0b0b"            # основной текст
INK_SECONDARY = "#52514e"  # подписи делений, второстепенный текст
GRID = "#e4e3df"           # сетка и вспомогательные линии
REFERENCE = "#8a8984"      # опорные линии (теория, пороги)


def apply_style() -> None:
    """Включает стиль проекта для всех следующих рисунков matplotlib."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=SERIES_COLORS),
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Сохраняет рисунок в PNG (папка создаётся при необходимости) и закрывает его."""
    path = resolve_path(path)
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    return path


def show_image_grid(
    images: torch.Tensor,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    title: str | None = None,
    cell_size: float = 1.0,
) -> plt.Figure:
    """Сетка картинок: строки — объекты, столбцы — варианты (например, шаги зашумления).

    Args:
        images: тензор [R, C, 3, H, W] с яркостями в [−1, 1].
        row_labels: подписи строк слева (R штук).
        col_labels: подписи столбцов сверху (C штук), можно в две строки через «\\n».
        title: общий заголовок.
        cell_size: размер одной картинки в дюймах.
    """
    n_rows, n_cols = images.shape[:2]
    # Место под подписи задаётся в дюймах, чтобы между заголовком и картинками не было пустоты.
    label_w = 1.0 if row_labels is not None else 0.0
    col_h = 0.45 if col_labels is not None else 0.0
    title_h = 0.4 if title else 0.0
    width = n_cols * cell_size + label_w
    height = n_rows * cell_size + col_h + title_h
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=label_w / width, right=1.0, bottom=0.0,
                        top=1.0 - (col_h + title_h) / height, wspace=0.04, hspace=0.04)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i][j]
            img = to_unit_range(images[i, j].detach().float().cpu()).permute(1, 2, 0).numpy()
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0 and col_labels is not None:
                ax.set_title(col_labels[j], fontsize=9, fontweight="normal")
            if j == 0 and row_labels is not None:
                ax.set_ylabel(row_labels[i], rotation=0, ha="right", va="center", fontsize=9)
    if title:
        fig.suptitle(title, y=1.0, va="top", fontsize=11, fontweight="bold")
    return fig
