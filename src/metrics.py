"""Метрики качества восстановления картинок: PSNR и SSIM, плюс доверительный интервал среднего.

Обе метрики сравнивают картинку с эталоном попиксельно и считаются для каждой картинки
батча отдельно. Картинки подаются с яркостями в [0, 1] (``data_range = 1``).

* PSNR (пиковое отношение сигнал/шум) = 10 · log10(1 / MSE), в децибелах. Чем больше, тем ближе
  к эталону; +6 дБ соответствуют уменьшению средней ошибки по амплитуде вдвое.
* SSIM (индекс структурного сходства, Wang et al., 2004) сравнивает яркость, контраст и структуру
  в скользящем гауссовом окне 11×11 (σ = 1.5) и лежит в [−1, 1]; 1 — полное совпадение.
  Лучше, чем PSNR, отражает то, как сходство воспринимает человек.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """PSNR каждой пары картинок, дБ. Для совпадающих картинок — бесконечность.

    Args:
        x, y: картинки [B, C, H, W] одной формы.
        data_range: размах яркостей (1 для [0, 1]).

    Returns:
        Тензор [B].
    """
    if x.shape != y.shape:
        raise ValueError(f"Формы не совпадают: {tuple(x.shape)} и {tuple(y.shape)}")
    mse = (x.double() - y.double()).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """Нормированное двумерное гауссово окно, повторённое для каждого канала: [C, 1, size, size]."""
    coords = torch.arange(size, dtype=torch.float64, device=device) - (size - 1) / 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g).expand(channels, 1, size, size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """SSIM каждой пары картинок, среднее по каналам и позициям окна.

    Локальные средние μ, дисперсии σ² и ковариация σ_xy считаются гауссовым окном только там,
    где окно целиком помещается в картинку (для 32×32 и окна 11 — область 22×22):

        SSIM = (2 μ_x μ_y + C1)(2 σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)),
        C1 = (0.01 · L)²,  C2 = (0.03 · L)²,  L = data_range.

    Совпадает с ``skimage.metrics.structural_similarity(..., gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False)``.

    Returns:
        Тензор [B].
    """
    if x.shape != y.shape:
        raise ValueError(f"Формы не совпадают: {tuple(x.shape)} и {tuple(y.shape)}")
    if min(x.shape[-2:]) < win_size:
        raise ValueError(f"Картинка меньше окна {win_size}×{win_size}")
    x, y = x.double(), y.double()
    channels = x.shape[1]
    window = _gaussian_window(win_size, sigma, channels, x.device)

    def blur(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, window, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov_xy = blur(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def mean_ci(values: torch.Tensor, z: float = 1.96) -> tuple[float, float]:
    """Среднее и полуширина 95%-го доверительного интервала: z · std / sqrt(n).

    Для сотен картинок распределение среднего близко к нормальному, поэтому достаточно
    нормального приближения.
    """
    values = values.double().flatten()
    n = values.numel()
    if n < 2:
        return float(values.mean()), float("nan")
    return float(values.mean()), float(z * values.std(unbiased=True) / math.sqrt(n))
