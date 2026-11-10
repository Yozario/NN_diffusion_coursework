"""Тесты метрик PSNR, SSIM и доверительного интервала."""

import math

import numpy as np
import pytest
import torch

from src.metrics import mean_ci, psnr, ssim
from src.utils import make_generator


@pytest.fixture
def images():
    gen = make_generator(0)
    x = torch.rand(4, 3, 32, 32, generator=gen)
    noisy = (x + 0.1 * torch.randn(x.shape, generator=gen)).clamp(0, 1)
    return x, noisy


def test_psnr_formula(images):
    x, noisy = images
    mse = ((x.double() - noisy.double()) ** 2).flatten(1).mean(1)
    assert torch.allclose(psnr(x, noisy), 10 * torch.log10(1 / mse))
    assert torch.isinf(psnr(x, x)).all()


def test_psnr_known_value():
    x = torch.zeros(1, 3, 32, 32)
    y = torch.full_like(x, 0.1)                      # MSE = 0.01 -> 20 дБ
    assert psnr(x, y).item() == pytest.approx(20.0)


def test_ssim_identity_and_symmetry(images):
    x, noisy = images
    assert torch.allclose(ssim(x, x), torch.ones(4, dtype=torch.float64))
    assert torch.allclose(ssim(x, noisy), ssim(noisy, x))


def test_ssim_decreases_with_noise():
    gen = make_generator(1)
    x = torch.rand(8, 3, 32, 32, generator=gen)
    noise = torch.randn(x.shape, generator=gen)
    values = [ssim(x, (x + s * noise).clamp(0, 1)).mean().item() for s in (0.02, 0.1, 0.3)]
    assert values[0] > values[1] > values[2]


def test_ssim_matches_skimage(images):
    """Сверка с эталонной реализацией (тест пропускается, если scikit-image не установлен)."""
    metrics = pytest.importorskip("skimage.metrics")
    x, noisy = images
    ours = ssim(x, noisy)
    for i in range(x.shape[0]):
        ref = metrics.structural_similarity(
            x[i].permute(1, 2, 0).numpy().astype(np.float64), noisy[i].permute(1, 2, 0).numpy().astype(np.float64),
            data_range=1.0, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        assert ours[i].item() == pytest.approx(ref, abs=1e-6)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        psnr(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 16, 16))
    with pytest.raises(ValueError):
        ssim(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))


def test_mean_ci():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mean, half = mean_ci(values)
    assert mean == 2.5
    assert half == pytest.approx(1.96 * values.std().item() / 2)
    assert math.isnan(mean_ci(torch.tensor([5.0]))[1])
