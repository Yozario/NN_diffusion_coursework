"""Тесты подготовки данных. Не требуют скачивания CIFAR-10."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import get_transforms, stratified_split, to_unit_range


@pytest.fixture
def targets():
    # 10 классов по 5000 объектов, перемешанные, — как в обучающей части CIFAR-10
    return np.random.default_rng(0).permutation(np.repeat(np.arange(10), 5000))


def test_split_sizes_and_disjoint(targets):
    train_idx, val_idx = stratified_split(targets, val_size=5000, seed=42)
    assert len(train_idx) == 45_000 and len(val_idx) == 5000
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.union1d(train_idx, val_idx)) == len(targets)


def test_split_is_stratified(targets):
    train_idx, val_idx = stratified_split(targets, val_size=5000, seed=42)
    assert np.all(np.bincount(targets[val_idx]) == 500)
    assert np.all(np.bincount(targets[train_idx]) == 4500)


def test_split_reproducible(targets):
    a = stratified_split(targets, 5000, seed=42)
    b = stratified_split(targets, 5000, seed=42)
    c = stratified_split(targets, 5000, seed=7)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_split_rejects_uneven_size(targets):
    with pytest.raises(ValueError):
        stratified_split(targets, val_size=5005, seed=0)


def test_transform_range():
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, 16:] = 255
    x = get_transforms(train=False)(Image.fromarray(pixels))
    assert x.shape == (3, 32, 32)
    assert x.min().item() == -1.0 and x.max().item() == 1.0
    assert torch.allclose(to_unit_range(x).mean(), torch.tensor(0.5))
