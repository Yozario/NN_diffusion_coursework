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


# --------------------------------------------------------------------------- #
# Обратный процесс
# --------------------------------------------------------------------------- #
class OracleDenoiser(torch.nn.Module):
    """Идеальная сеть для данных из одной картинки x*: возвращает точный шум, которым x* превратилась в x_t.

    Если обратный процесс реализован верно, с такой «сетью» генерация из любого шума
    обязана прийти ровно в x*.
    """

    def __init__(self, diffusion, target):
        super().__init__()
        self.diffusion = diffusion
        self.register_buffer("target", target)
        self.dummy = torch.nn.Parameter(torch.zeros(()))  # чтобы у модели было устройство

    def forward(self, x_t, t):
        abar = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
        return (x_t - abar.sqrt() * self.target) / (1 - abar).sqrt()


@pytest.fixture
def target():
    return torch.rand(2, 3, 4, 4, generator=make_generator(5)) * 1.6 - 0.8


@pytest.mark.parametrize("name", ["linear", "cosine"])
def test_posterior_coefficients(name):
    """Коэффициенты сверяются с расчётом в float64 напрямую по β_t."""
    betas64 = make_beta_schedule(name, T)
    diffusion = GaussianDiffusion(betas64)
    beta = torch.cat([torch.zeros(1, dtype=torch.float64), betas64])
    abar = torch.cumprod(1 - beta, dim=0)
    for t in [2, 10, 500, T]:
        c1 = abar[t - 1].sqrt() * beta[t] / (1 - abar[t])
        c2 = (1 - beta[t]).sqrt() * (1 - abar[t - 1]) / (1 - abar[t])
        var = beta[t] * (1 - abar[t - 1]) / (1 - abar[t])
        assert diffusion.posterior_mean_coef1[t].item() == pytest.approx(c1.item(), rel=1e-5)
        assert diffusion.posterior_mean_coef2[t].item() == pytest.approx(c2.item(), rel=1e-5)
        assert diffusion.posterior_variance[t].item() == pytest.approx(var.item(), rel=1e-5)
    assert diffusion.posterior_variance[1].item() == 0.0       # последний шаг без шума
    assert diffusion.posterior_mean_coef1[1].item() == pytest.approx(1.0, abs=1e-5)
    assert diffusion.posterior_mean_coef2[1].item() == pytest.approx(0.0, abs=1e-6)


def test_posterior_mean_matches_paper_formula(diffusion):
    """Через x̂_0 (без обрезки) получается та же формула, что в статье: (x_t − β_t/sqrt(1−ᾱ_t)·ε) / sqrt(α_t)."""
    x_t, eps = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    for t in [2, 100, 900]:
        mean = diffusion.q_posterior_mean(diffusion.predict_x0(x_t, t, eps), x_t, t)
        paper = (x_t - diffusion.betas[t] / (1 - diffusion.alphas_cumprod[t]).sqrt() * eps) / diffusion.alphas[t].sqrt()
        assert torch.allclose(mean, paper, atol=1e-4)


def test_predict_x0_inverts_q_sample(diffusion, target):
    noise = torch.randn_like(target)
    for t in [1, 100, 700]:
        x_t = diffusion.q_sample(target, t, noise=noise)
        assert torch.allclose(diffusion.predict_x0(x_t, t, noise), target, atol=1e-3)


@pytest.mark.parametrize("variance", ["beta", "posterior"])
def test_sampling_with_oracle_reaches_target(diffusion, target, variance):
    out = diffusion.p_sample_loop(OracleDenoiser(diffusion, target), shape=target.shape,
                                  generator=make_generator(0), variance=variance)
    assert torch.allclose(out.images, target, atol=1e-4)


def test_reconstruct_with_oracle(diffusion, target):
    oracle = OracleDenoiser(diffusion, target)
    assert torch.equal(diffusion.reconstruct(oracle, target, 0).images, target)
    assert torch.allclose(diffusion.reconstruct(oracle, target, 500, generator=make_generator(1)).images,
                          target, atol=1e-4)


def test_sampling_records_trajectory(diffusion, target):
    oracle = OracleDenoiser(diffusion, target)
    out = diffusion.p_sample_loop(oracle, shape=target.shape, generator=make_generator(0),
                                  record_steps=[T, 500, 0])
    assert set(out.x_t) == {T, 500, 0}
    assert set(out.x0_pred) == {T, 500}                  # на шаге 0 сеть уже не вызывается
    assert torch.equal(out.x_t[0], out.images.cpu())
    assert out.x_t[T].shape == target.shape


def test_sampling_reproducible_and_restores_mode(diffusion, target):
    oracle = OracleDenoiser(diffusion, target).train()
    a = diffusion.p_sample_loop(oracle, shape=target.shape, t_start=50, generator=make_generator(3),
                                clip_x0=False, record_steps=[25])
    b = diffusion.p_sample_loop(oracle, shape=target.shape, t_start=50, generator=make_generator(3),
                                clip_x0=False, record_steps=[25])
    assert torch.equal(a.x_t[25], b.x_t[25])
    assert oracle.training                               # режим сети вернулся


def test_p_sample_rejects_bad_arguments(diffusion, target):
    oracle = OracleDenoiser(diffusion, target)
    with pytest.raises(ValueError):
        diffusion.p_sample(oracle, target, 0)
    with pytest.raises(ValueError):
        diffusion.p_sample(oracle, target, 10, variance="learned")
    with pytest.raises(ValueError):
        diffusion.p_sample_loop(oracle)


def test_training_loss(diffusion, target):
    gen = make_generator(0)
    oracle_loss = diffusion.training_loss(OracleDenoiser(diffusion, target), target, generator=gen)
    assert oracle_loss.item() < 1e-6                     # идеальная сеть — нулевая потеря

    class Zero(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    x0 = torch.zeros(256, 3, 8, 8)
    zero_loss = diffusion.training_loss(Zero(), x0, generator=gen)
    assert zero_loss.item() == pytest.approx(1.0, abs=0.02)  # нулевой ответ — потеря E[ε²] = 1
