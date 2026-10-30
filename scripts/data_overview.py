"""Обзор датасета CIFAR-10 для раздела «Данные» отчёта.

Что получается на выходе (папки задаются в configs/default.yaml):
    results/data_stats.csv             — число картинок каждого класса в train / val / test
    results/data_pixel_stats.json      — яркости по каналам после нормализации в [−1, 1]
    results/figures/data_samples.png   — по 8 примеров каждого класса

Запуск из корня репозитория:
    python -m scripts.data_overview
    python -m scripts.data_overview --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from src.data import CIFAR10_CLASSES, CIFAR10_CLASSES_RU, NUM_CLASSES, load_cifar10_from_config, subset_targets
from src.plotting import apply_style, save_figure, show_image_grid
from src.utils import ensure_dir, get_logger, load_config, resolve_path, save_json, set_seed

SAMPLES_PER_CLASS = 8


def class_counts(splits) -> dict[str, np.ndarray]:
    """Число объектов каждого класса в каждой части датасета."""
    return {
        name: np.bincount(subset_targets(getattr(splits, name)), minlength=NUM_CLASSES)
        for name in ("train", "val", "test")
    }


def save_counts_csv(counts: dict[str, np.ndarray], path) -> None:
    """Таблица «класс × часть датасета» с итоговой строкой."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "class_ru", "train", "val", "test"])
        for c in range(NUM_CLASSES):
            writer.writerow([CIFAR10_CLASSES[c], CIFAR10_CLASSES_RU[c],
                             counts["train"][c], counts["val"][c], counts["test"][c]])
        writer.writerow(["total", "всего", counts["train"].sum(), counts["val"].sum(), counts["test"].sum()])


def pixel_stats(splits) -> dict[str, dict[str, list[float]]]:
    """Среднее, стандартное отклонение, минимум и максимум яркости по каналам R, G, B.

    Считается по обучающей части после перевода в [−1, 1] (без аугментаций):
    подтверждает, что данные лежат в нужном диапазоне и примерно центрированы.
    Картинки обрабатываются частями по 5000, чтобы не занимать гигабайт памяти.
    """
    base = splits.train.dataset                      # CIFAR10 со всеми 50 000 картинками
    indices = np.asarray(splits.train.indices)
    total, total_sq, count = np.zeros(3), np.zeros(3), 0
    lo, hi = np.full(3, 255), np.zeros(3)
    for start in range(0, len(indices), 5000):
        pixels = base.data[indices[start:start + 5000]]          # uint8 [n, 32, 32, 3]
        lo = np.minimum(lo, pixels.min(axis=(0, 1, 2)))
        hi = np.maximum(hi, pixels.max(axis=(0, 1, 2)))
        x = pixels.astype(np.float64) / 255.0 * 2.0 - 1.0
        total += x.sum(axis=(0, 1, 2))
        total_sq += (x ** 2).sum(axis=(0, 1, 2))
        count += x.shape[0] * x.shape[1] * x.shape[2]
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return {
        "channels": ["R", "G", "B"],
        "mean": mean.round(4).tolist(),
        "std": std.round(4).tolist(),
        "min": (lo / 255.0 * 2.0 - 1.0).tolist(),
        "max": (hi / 255.0 * 2.0 - 1.0).tolist(),
    }


def sample_grid(dataset, seed: int) -> torch.Tensor:
    """По SAMPLES_PER_CLASS случайных картинок каждого класса: тензор [10, 8, 3, 32, 32]."""
    rng = np.random.default_rng(seed)
    targets = subset_targets(dataset)
    rows = []
    for c in range(NUM_CLASSES):
        idx = rng.choice(np.flatnonzero(targets == c), size=SAMPLES_PER_CLASS, replace=False)
        rows.append(torch.stack([dataset[int(i)][0] for i in idx]))
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="путь к конфигу")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    paths = cfg["paths"]
    logger = get_logger("data_overview", log_file=resolve_path(paths["logs_dir"]) / "data_overview.log")
    results_dir = ensure_dir(paths["results_dir"])
    figures_dir = ensure_dir(paths["figures_dir"])
    apply_style()

    logger.info("Загружаю CIFAR-10 (при первом запуске скачивается ~170 МБ)")
    splits = load_cifar10_from_config(cfg)

    counts = class_counts(splits)
    save_counts_csv(counts, results_dir / "data_stats.csv")
    logger.info("Размеры частей: train %d, val %d, test %d",
                counts["train"].sum(), counts["val"].sum(), counts["test"].sum())
    for c in range(NUM_CLASSES):
        logger.info("  %-10s train %5d | val %4d | test %5d", CIFAR10_CLASSES_RU[c],
                    counts["train"][c], counts["val"][c], counts["test"][c])

    stats = pixel_stats(splits)
    save_json(stats, results_dir / "data_pixel_stats.json")
    logger.info("Яркости после нормализации (R, G, B): среднее %s, ст. откл. %s, диапазон [%.1f, %.1f]",
                stats["mean"], stats["std"], min(stats["min"]), max(stats["max"]))

    grid = sample_grid(splits.val, seed=cfg["seed"])
    fig = show_image_grid(grid, row_labels=CIFAR10_CLASSES_RU,
                          title="Примеры картинок CIFAR-10 по классам")
    path = save_figure(fig, figures_dir / "data_samples.png")
    logger.info("Рисунок сохранён: %s", path)


if __name__ == "__main__":
    main()
