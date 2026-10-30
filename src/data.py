"""Загрузка и подготовка датасета CIFAR-10.

Что делает модуль:

* скачивает CIFAR-10 (50 000 обучающих и 10 000 тестовых картинок 32×32, 10 классов);
* переводит яркости пикселей из [0, 1] в [−1, 1] — в этом диапазоне работает
  диффузионная модель: чистый шум N(0, I) и данные имеют сопоставимый масштаб;
* для обучающей части добавляет случайное горизонтальное отражение;
* делит официальную обучающую выборку на train (45 000) и val (5 000)
  стратифицированно, то есть с одинаковой долей каждого класса, и с фиксированным seed.

Тестовая выборка (10 000) остаётся нетронутой.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from src.utils import make_generator, resolve_path, seed_worker

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR10_CLASSES_RU = (
    "самолёт", "автомобиль", "птица", "кошка", "олень",
    "собака", "лягушка", "лошадь", "корабль", "грузовик",
)
NUM_CLASSES = len(CIFAR10_CLASSES)


# --------------------------------------------------------------------------- #
# Преобразования
# --------------------------------------------------------------------------- #
def get_transforms(train: bool, hflip: bool = True) -> T.Compose:
    """Преобразования картинки: PIL → тензор [3, 32, 32] с яркостями в [−1, 1].

    ``Normalize(mean=0.5, std=0.5)`` делает (x − 0.5) / 0.5 = 2x − 1, то есть [0, 1] → [−1, 1].
    Отражение применяется только при ``train=True``: валидация и тест должны быть детерминированными.
    """
    ops: list[Any] = []
    if train and hflip:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    return T.Compose(ops)


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Обратное преобразование для отображения: [−1, 1] → [0, 1] с обрезкой выбросов."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Разбиение train / val
# --------------------------------------------------------------------------- #
def stratified_split(targets: Sequence[int], val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Делит индексы на train и val так, чтобы в val было поровну картинок каждого класса.

    Args:
        targets: метки классов всех объектов.
        val_size: размер валидационной части; должен делиться на число классов.
        seed: seed перемешивания — одно и то же значение даёт одно и то же разбиение.

    Returns:
        Отсортированные массивы индексов ``(train_idx, val_idx)``.
    """
    targets = np.asarray(targets)
    classes = np.unique(targets)
    if val_size % len(classes) != 0:
        raise ValueError(f"val_size={val_size} должен делиться на число классов ({len(classes)})")
    per_class = val_size // len(classes)

    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for c in classes:
        idx = np.flatnonzero(targets == c)
        if per_class >= len(idx):
            raise ValueError(f"В классе {c} всего {len(idx)} объектов, а в val нужно {per_class}")
        idx = rng.permutation(idx)
        val_parts.append(idx[:per_class])
        train_parts.append(idx[per_class:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def subset_targets(dataset: Dataset) -> np.ndarray:
    """Метки классов для CIFAR10 или Subset над ним (без загрузки самих картинок)."""
    if isinstance(dataset, Subset):
        return np.asarray(subset_targets(dataset.dataset))[dataset.indices]
    return np.asarray(dataset.targets)


# --------------------------------------------------------------------------- #
# Датасеты и загрузчики
# --------------------------------------------------------------------------- #
@dataclass
class CifarSplits:
    """Три части датасета, все с яркостями в [−1, 1]."""

    train: Dataset  # 45 000, со случайным отражением
    val: Dataset    # 5 000, без аугментаций
    test: Dataset   # 10 000, без аугментаций


def load_cifar10(
    data_dir: str | Path = "data",
    val_size: int = 5000,
    seed: int = 42,
    hflip: bool = True,
    download: bool = True,
) -> CifarSplits:
    """Загружает CIFAR-10 и возвращает части train / val / test.

    Train и val берутся из одного файла официальной обучающей выборки, но с разными
    преобразованиями. Чтобы не держать картинки в памяти дважды, второй объект —
    поверхностная копия первого: массив пикселей общий, отличается только ``transform``.
    """
    root = resolve_path(data_dir)
    train_full = CIFAR10(root=str(root), train=True, download=download,
                         transform=get_transforms(train=True, hflip=hflip))
    train_plain = copy.copy(train_full)
    train_plain.transform = get_transforms(train=False)
    test = CIFAR10(root=str(root), train=False, download=download,
                   transform=get_transforms(train=False))

    train_idx, val_idx = stratified_split(train_full.targets, val_size=val_size, seed=seed)
    return CifarSplits(
        train=Subset(train_full, train_idx.tolist()),
        val=Subset(train_plain, val_idx.tolist()),
        test=test,
    )


def load_cifar10_from_config(cfg: dict[str, Any], download: bool = True) -> CifarSplits:
    """То же, что :func:`load_cifar10`, но параметры берутся из конфига проекта."""
    return load_cifar10(
        data_dir=cfg["paths"]["data_dir"],
        val_size=cfg["data"]["val_size"],
        seed=cfg["seed"],
        hflip=cfg["data"]["hflip"],
        download=download,
    )


def make_dataloaders(cfg: dict[str, Any], splits: CifarSplits | None = None) -> dict[str, DataLoader]:
    """Загрузчики батчей для train / val / test с воспроизводимым порядком.

    Порядок перемешивания train задаёт отдельный генератор с seed из конфига,
    поэтому при перезапуске батчи идут в той же последовательности.
    ``drop_last=True`` для train, чтобы все батчи были одного размера.
    """
    if splits is None:
        splits = load_cifar10_from_config(cfg)
    dcfg = cfg["data"]
    num_workers = dcfg["num_workers"]
    common = dict(
        batch_size=dcfg["batch_size"],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )
    return {
        "train": DataLoader(splits.train, shuffle=True, drop_last=True,
                            generator=make_generator(cfg["seed"]), **common),
        "val": DataLoader(splits.val, shuffle=False, **common),
        "test": DataLoader(splits.test, shuffle=False, **common),
    }


def fixed_examples(
    dataset: Dataset,
    n: int,
    seed: int,
    one_per_class: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Фиксированный набор картинок для визуализаций (одинаковый при каждом запуске).

    При ``one_per_class=True`` берётся по одной случайной картинке из первых ``n`` классов,
    иначе — ``n`` случайных картинок. Используется в экспериментах 1 и 2,
    чтобы на всех рисунках были одни и те же объекты.
    Передавать val или test: в train картинки случайно отражаются при каждом обращении.

    Returns:
        ``(images, labels)``: тензор [n, 3, 32, 32] в [−1, 1] и метки [n].
    """
    rng = np.random.default_rng(seed)
    targets = subset_targets(dataset)
    if one_per_class:
        if n > NUM_CLASSES:
            raise ValueError(f"По одной на класс можно взять не больше {NUM_CLASSES} картинок")
        indices = [int(rng.choice(np.flatnonzero(targets == c))) for c in range(n)]
    else:
        indices = rng.choice(len(dataset), size=n, replace=False).tolist()

    images, labels = zip(*(dataset[i] for i in indices))
    return torch.stack(images), torch.tensor(labels)
