"""Тесты оценки генерации: преобразования картинок, FID по признакам, ближайшие соседи."""

import numpy as np
import pytest
import torch

from src.fid import fid_from_features, frechet_distance, from_uint8, nearest_neighbors, to_uint8


@pytest.fixture
def uint8_images():
    return np.random.default_rng(0).integers(0, 256, size=(40, 32, 32, 3), dtype=np.uint8)


def test_uint8_roundtrip(uint8_images):
    x = from_uint8(uint8_images)
    assert x.shape == (40, 3, 32, 32)
    assert x.min() >= -1 and x.max() <= 1
    assert np.array_equal(to_uint8(x), uint8_images)


def test_to_uint8_clips():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 5.0]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    assert to_uint8(x)[0, 0, :, 0].tolist() == [0, 0, 128, 255, 255]


def test_fid_identical_and_shifted():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2000, 8))
    b = rng.normal(size=(2000, 8))
    assert fid_from_features(a, a) == pytest.approx(0.0, abs=1e-6)
    near = fid_from_features(a, b)                   # та же выборка из того же распределения
    far = fid_from_features(a, b + 1.0)              # сдвиг среднего на 1 по 8 осям
    assert near < 0.1
    assert far == pytest.approx(8.0 + near, abs=0.2)  # ‖Δμ‖² = 8


def test_nearest_neighbors_matches_brute_force(uint8_images):
    query = uint8_images[:7]
    reference = uint8_images[5:]                     # первые две картинки запроса есть в reference
    d, i = nearest_neighbors(query, reference, k=3, query_chunk=4, reference_chunk=9)

    q = query.reshape(7, -1).astype(np.float64) / 255
    r = reference.reshape(len(reference), -1).astype(np.float64) / 255
    full = np.sqrt(((q[:, None] - r[None]) ** 2).mean(axis=2))
    expected_i = np.argsort(full, axis=1)[:, :3]
    assert np.array_equal(i, expected_i)
    assert np.allclose(d, np.take_along_axis(full, expected_i, axis=1), atol=1e-12)
    assert d[5, 0] == 0.0 and i[5, 0] == 0           # картинка 5 запроса — это reference[0]


def test_frechet_distance_matches_closed_form():
    """Для диагональных ковариаций корень считается по диагонали: tr(Σ1 + Σ2 − 2 sqrt(Σ1 Σ2))."""
    s1, s2 = np.array([1.0, 4.0, 9.0]), np.array([4.0, 1.0, 1.0])
    mu1, mu2 = np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 0.0])
    expected = 5.0 + (s1 + s2 - 2 * np.sqrt(s1 * s2)).sum()
    assert frechet_distance(mu1, np.diag(s1), mu2, np.diag(s2)) == pytest.approx(expected)
