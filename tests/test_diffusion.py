"""Тесты прямого процесса: расписания шума, замкнутая формула, пошаговый процесс."""

import pytest
import torch
from scipy import stats

from src.diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule, make_beta_schedule
from src.utils import make_generator

T = 1000


@pytest.fixture(params=["linear", "cosine"])
def diffusion(request):
    return GaussianDiffusion(make_beta_schedule(request.param, T))


# --------------------------------------------------------------------------- #
# Расписания
# --------------------------------------------------------------------------- #
def test_linear_schedule_endpoints():
    betas = linear_beta_schedule(T, 1e-4, 0.02)
    assert betas.shape == (T,)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)
    assert torch.all(betas[1:] > betas[:-1])


def test_cosine_schedule_valid():
    betas = cosine_beta_schedule(T)
    assert betas.shape == (T,)
    assert torch.all((betas > 0) & (betas <= 0.999))


def test_unknown_schedule():
    with pytest.raises(ValueError):
        make_beta_schedule("quadratic", T)


def test_coefficients_indexing(diffusion):
    """Индекс массива = номер шага: на шаге 0 шума нет, дальше сигнал монотонно убывает."""
    abar = diffusion.alphas_cumprod
    assert abar.shape == (T + 1,)
    assert abar[0].item() == 1.0
    assert diffusion.betas[0].item() == 0.0
    assert torch.all(abar[1:] < abar[:-1])
    assert abar[-1].item() < 1e-3
    assert torch.isinf(diffusion.snr(0))


def test_coefficients_follow_device(diffusion):
    """Буферы — часть модуля; ``.double()`` меняет их тип так же, как ``.to(device)`` переносит."""
    assert diffusion.double().alphas_cumprod.dtype == torch.float64


# --------------------------------------------------------------------------- #
# Замкнутая формула и пошаговый процесс
# --------------------------------------------------------------------------- #
def test_q_sample_t0_returns_x0(diffusion):
    x0 = torch.rand(4, 3, 8, 8) * 2 - 1
    assert torch.equal(diffusion.q_sample(x0, 0), x0)
    assert torch.equal(diffusion.q_sample_iterative(x0, 0), x0)


def test_q_sample_formula(diffusion):
    x0 = torch.rand(2, 3, 4, 4) * 2 - 1
    noise = torch.randn_like(x0)
    t = torch.tensor([10, 700])
    expected = torch.stack([
        diffusion.alphas_cumprod[k].sqrt() * x0[i] + (1 - diffusion.alphas_cumprod[k]).sqrt() * noise[i]
        for i, k in enumerate(t)
    ])
    assert torch.allclose(diffusion.q_sample(x0, t, noise=noise), expected, atol=1e-6)


def test_q_step_formula(diffusion):
    x = torch.rand(1, 3, 4, 4)
    noise = torch.randn_like(x)
    expected = diffusion.alphas[5].sqrt() * x + diffusion.betas[5].sqrt() * noise
    assert torch.allclose(diffusion.q_step(x, 5, noise=noise), expected, atol=1e-7)


def test_timestep_range_checked(diffusion):
    x0 = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError):
        diffusion.q_sample(x0, T + 1)
    with pytest.raises(ValueError):
        diffusion.q_sample(x0, -1)


def test_iterative_respects_per_sample_t(diffusion):
    """Каждая картинка батча проходит ровно свои t шагов."""
    x0 = torch.rand(3, 3, 4, 4)
    t = torch.tensor([0, 3, 7])
    gen = make_generator(0)
    x_t, saved = diffusion.q_sample_iterative(x0, t, generator=gen, record=[3, 7])
    assert torch.equal(x_t[0], x0[0])               # t = 0 — без изменений
    assert torch.equal(x_t[1], saved[3][1])         # после 3-го шага картинка 1 заморожена
    assert torch.equal(x_t[2], saved[7][2])


def test_iterative_is_reproducible(diffusion):
    x0 = torch.rand(2, 3, 4, 4)
    a = diffusion.q_sample_iterative(x0, 50, generator=make_generator(1))
    b = diffusion.q_sample_iterative(x0, 50, generator=make_generator(1))
    assert torch.equal(a, b)


@pytest.mark.parametrize("t", [1, 50, 400, T])
def test_iterative_matches_closed_form_distribution(diffusion, t):
    """Пошаговый процесс даёт то же распределение, что и замкнутая формула.

    Стандартизованный остаток z = (x_t − sqrt(ᾱ_t)·x_0) / sqrt(1 − ᾱ_t) должен быть N(0, 1).
    """
    x0 = torch.rand(64, 3, 8, 8) * 2 - 1
    abar = diffusion.alphas_cumprod[t].double()
    x_t = diffusion.q_sample_iterative(x0, t, generator=make_generator(123)).double()
    z = ((x_t - abar.sqrt() * x0) / (1 - abar).sqrt()).flatten().numpy()
    assert abs(z.mean()) < 0.02
    assert abs(z.std() - 1) < 0.02
    assert stats.kstest(z, "norm").pvalue > 1e-3


def test_sample_timesteps_range(diffusion):
    t = diffusion.sample_timesteps(10_000, generator=make_generator(0))
    assert t.min() == 1 and t.max() == T
