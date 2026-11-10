"""Тесты U-Net: формы, кодирование шага, инициализация, связность графа."""

import pytest
import torch

from src.unet import UNet, count_parameters, timestep_embedding
from src.utils import load_config


@pytest.fixture
def small_unet():
    """Уменьшенная сеть той же структуры — чтобы тесты на CPU шли быстро."""
    torch.manual_seed(0)
    return UNet(base_channels=32, channel_mults=(1, 2), num_res_blocks=1, attn_resolutions=(16,),
                num_heads=2, dropout=0.0)


def perturb(model):
    """Заменяет нулевую инициализацию случайной, чтобы выход сети зависел от входа."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return model


# --------------------------------------------------------------------------- #
# Кодирование шага
# --------------------------------------------------------------------------- #
def test_timestep_embedding_shape_and_range():
    emb = timestep_embedding(torch.tensor([0, 1, 500, 1000]), 64)
    assert emb.shape == (4, 64)
    assert emb.abs().max() <= 1.0


def test_timestep_embedding_distinguishes_steps():
    emb = timestep_embedding(torch.arange(1, 1001), 64)
    dist = torch.cdist(emb, emb)
    dist.fill_diagonal_(float("inf"))
    assert dist.min() > 1e-3  # у всех 1000 шагов разные векторы


def test_timestep_embedding_requires_even_dim():
    with pytest.raises(ValueError):
        timestep_embedding(torch.tensor([1]), 63)


# --------------------------------------------------------------------------- #
# Сеть
# --------------------------------------------------------------------------- #
def test_config_model_size():
    """Сеть из конфига проекта — 6–9 млн параметров, как запланировано под 4 ГБ видеопамяти."""
    n = count_parameters(UNet.from_config(load_config()))
    assert 6_000_000 < n < 9_000_000


def test_output_shape(small_unet):
    x = torch.randn(3, 3, 32, 32)
    assert small_unet(x, torch.tensor([1, 500, 1000])).shape == x.shape
    assert small_unet(x, 250).shape == x.shape  # один шаг на весь батч


def test_zero_output_at_init(small_unet):
    """Выходной слой инициализирован нулями: до обучения сеть предсказывает нулевой шум."""
    assert torch.count_nonzero(small_unet(torch.randn(2, 3, 32, 32), 10)) == 0


def test_output_depends_on_timestep(small_unet):
    model = perturb(small_unet).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(model(x, 10), model(x, 900))


def test_samples_in_batch_are_independent(small_unet):
    """GroupNorm не смешивает объекты батча: выход для картинки не зависит от соседей."""
    model = perturb(small_unet).eval()
    x = torch.randn(4, 3, 32, 32)
    t = torch.tensor([5, 50, 500, 999])
    with torch.no_grad():
        together = model(x, t)
        alone = torch.cat([model(x[i:i + 1], t[i:i + 1]) for i in range(4)])
    assert torch.allclose(together, alone, atol=1e-5)


def test_all_parameters_receive_gradient(small_unet):
    """В графе вычислений участвуют все параметры — нет «забытых» слоёв."""
    model = perturb(small_unet)
    model(torch.randn(2, 3, 32, 32), torch.tensor([3, 700])).pow(2).mean().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing


def test_rejects_bad_image_size():
    with pytest.raises(ValueError):
        UNet(image_size=30, channel_mults=(1, 2, 2))
