r"""Гауссовский диффузионный процесс (DDPM, Ho et al., 2020).

Обозначения совпадают со статьёй. x_0 — исходная картинка, x_t — картинка после t шагов
зашумления, t = 0, 1, ..., T. Один шаг прямого процесса:

    x_t = sqrt(1 − β_t) · x_{t−1} + sqrt(β_t) · ε,   ε ~ N(0, I).

Отсюда по индукции следует формула в замкнутом виде — x_t можно получить из x_0 сразу,
без прохода по всем промежуточным шагам:

    x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε,   α_t = 1 − β_t,   ᾱ_t = α_1 · α_2 · … · α_t.

Соглашение об индексах в коде: все массивы коэффициентов имеют длину T + 1, а индекс равен
номеру шага. Нулевой элемент соответствует отсутствию шума: β_0 = 0, ᾱ_0 = 1, поэтому
``q_sample(x0, t=0)`` возвращает саму x_0. При обучении шаг t выбирается из {1, ..., T}.

Модуль делят две роли:
    * прямой процесс (расписания шума, пошаговое зашумление, замкнутая формула) — роль 1;
    * обратный процесс (шаг генерации, полный цикл, сохранение траектории) — роль 2,
      добавляется в класс :class:`GaussianDiffusion` ниже.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Расписания шума β_1, ..., β_T
# --------------------------------------------------------------------------- #
def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Линейное расписание из статьи DDPM: β_t растёт равномерно от ``beta_start`` до ``beta_end``.

    Returns:
        Тензор float64 длины ``num_steps`` со значениями β_1, ..., β_T.
    """
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)


def cosine_beta_schedule(num_steps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Косинусное расписание (Nichol & Dhariwal, 2021, «Improved DDPM»).

    Задаётся сразу через ᾱ_t = f(t) / f(0), f(t) = cos²((t/T + s) / (1 + s) · π/2),
    а β_t восстанавливаются как 1 − ᾱ_t / ᾱ_{t−1}. Сигнал убывает медленнее, чем при линейном
    расписании, поэтому меньше шагов тратится на почти чистый шум. Последние β_t близки к 1,
    поэтому их ограничивают сверху значением ``max_beta``, как в оригинальной статье.

    Returns:
        Тензор float64 длины ``num_steps`` со значениями β_1, ..., β_T.
    """
    t = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=max_beta)


def make_beta_schedule(name: str, num_steps: int, **kwargs: Any) -> torch.Tensor:
    """Расписание по имени: ``"linear"`` или ``"cosine"``."""
    if name == "linear":
        return linear_beta_schedule(num_steps, **kwargs)
    if name == "cosine":
        return cosine_beta_schedule(num_steps, **kwargs)
    raise ValueError(f"Неизвестное расписание шума: {name!r} (ожидается 'linear' или 'cosine')")


# --------------------------------------------------------------------------- #
# Диффузионный процесс
# --------------------------------------------------------------------------- #
class GaussianDiffusion(nn.Module):
    """Коэффициенты диффузионного процесса и операции над ним.

    Наследуется от ``nn.Module`` только ради буферов: ``diffusion.to("cuda")`` переносит
    все коэффициенты на GPU вместе с моделью. Обучаемых параметров у класса нет.

    Коэффициенты считаются в float64 (произведение 1000 множителей теряет точность
    в float32), а хранятся в float32 — в том же типе, что и картинки.

    Args:
        betas: значения β_1, ..., β_T (без нулевого шага), например из :func:`make_beta_schedule`.
    """

    def __init__(self, betas: torch.Tensor) -> None:
        super().__init__()
        betas = torch.as_tensor(betas, dtype=torch.float64)
        if betas.ndim != 1 or not torch.all((betas > 0) & (betas < 1)):
            raise ValueError("betas должны быть одномерным тензором со значениями в (0, 1)")
        self.num_steps = len(betas)

        # Добавляем шаг t = 0 без шума, чтобы индекс совпадал с номером шага.
        betas = torch.cat([torch.zeros(1, dtype=torch.float64), betas])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        def register(name: str, value: torch.Tensor) -> None:
            self.register_buffer(name, value.to(torch.float32), persistent=False)

        register("betas", betas)                                        # β_t
        register("alphas", alphas)                                      # α_t = 1 − β_t
        register("alphas_cumprod", alphas_cumprod)                      # ᾱ_t
        register("sqrt_alphas", alphas.sqrt())                          # sqrt(α_t)   — шаг процесса
        register("sqrt_betas", betas.sqrt())                            # sqrt(β_t)   — шаг процесса
        register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())          # sqrt(ᾱ_t)   — замкнутая формула
        register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())  # sqrt(1 − ᾱ_t)
        register("snr_values", alphas_cumprod / (1.0 - alphas_cumprod))          # ᾱ_t / (1 − ᾱ_t), при t=0 — inf

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GaussianDiffusion":
        """Создаёт процесс по разделу ``diffusion`` конфига проекта."""
        dcfg = cfg["diffusion"]
        name = dcfg["schedule"]
        if name == "linear":
            kwargs = {"beta_start": dcfg["beta_start"], "beta_end": dcfg["beta_end"]}
        else:
            kwargs = {"s": dcfg["cosine_s"]}
        return cls(make_beta_schedule(name, dcfg["num_steps"], **kwargs))

    # ------------------------------------------------------------------ #
    # Вспомогательные функции
    # ------------------------------------------------------------------ #
    def _timesteps(self, t: int | torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Приводит номер шага к тензору long формы [B] и проверяет диапазон 0..T."""
        t = torch.as_tensor(t, device=device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.shape != (batch_size,):
            raise ValueError(f"t должен быть числом или тензором формы [{batch_size}], получено {tuple(t.shape)}")
        if t.min() < 0 or t.max() > self.num_steps:
            raise ValueError(f"t должен лежать в диапазоне 0..{self.num_steps}")
        return t

    @staticmethod
    def _extract(coeffs: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Берёт коэффициенты для шагов t и придаёт им форму [B, 1, 1, 1] для умножения на картинки."""
        return coeffs[t].view(-1, *([1] * (ndim - 1)))

    @staticmethod
    def _randn_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """Стандартный гауссов шум формы x; с генератором — воспроизводимый."""
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    # ------------------------------------------------------------------ #
    # Прямой процесс (роль 1)
    # ------------------------------------------------------------------ #
    def q_sample(
        self,
        x0: torch.Tensor,
        t: int | torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Замкнутая формула: x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε за одну операцию.

        Этой функцией пользуется обучение: для каждого объекта батча берётся свой
        случайный шаг t, а шум ε затем служит целью для U-Net.

        Args:
            x0: чистые картинки [B, C, H, W] в [−1, 1].
            t: номер шага — одно число для всего батча или тензор [B].
            noise: готовый шум ε той же формы, что x0; если не задан — генерируется.
            generator: генератор для воспроизводимого шума (если ``noise`` не задан).
        """
        t = self._timesteps(t, x0.shape[0], x0.device)
        if noise is None:
            noise = self._randn_like(x0, generator)
        return (self._extract(self.sqrt_alphas_cumprod, t, x0.ndim) * x0
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim) * noise)

    def q_step(
        self,
        x_prev: torch.Tensor,
        t: int | torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Один шаг прямого процесса: x_t = sqrt(α_t) · x_{t−1} + sqrt(β_t) · ε.

        При t = 0 возвращает вход без изменений (β_0 = 0).
        """
        t = self._timesteps(t, x_prev.shape[0], x_prev.device)
        if noise is None:
            noise = self._randn_like(x_prev, generator)
        return (self._extract(self.sqrt_alphas, t, x_prev.ndim) * x_prev
                + self._extract(self.sqrt_betas, t, x_prev.ndim) * noise)

    def q_sample_iterative(
        self,
        x0: torch.Tensor,
        t: int | torch.Tensor,
        generator: torch.Generator | None = None,
        record: Iterable[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Пошаговый прямой процесс: применяет :meth:`q_step` t раз подряд, начиная с x_0.

        Медленнее замкнутой формулы в t раз и нужен для проверки и визуализации: так зашумление
        выглядит «как в определении», и можно сохранить одну и ту же траекторию на разных шагах.

        Args:
            x0: чистые картинки [B, C, H, W].
            t: сколько шагов сделать — одно число или тензор [B] (у каждой картинки свой t).
            generator: генератор шума для воспроизводимости.
            record: номера шагов, на которых нужно сохранить копию текущего x
                (например, ``[0, 100, 500, 1000]``).

        Returns:
            x_t, а если задан ``record`` — пара ``(x_t, {шаг: x на этом шаге})``.
        """
        t = self._timesteps(t, x0.shape[0], x0.device)
        record_steps = set(record or [])
        if record_steps and (min(record_steps) < 0 or max(record_steps) > int(t.max())):
            raise ValueError(f"Шаги для записи должны лежать в 0..{int(t.max())}")
        saved: dict[int, torch.Tensor] = {}

        x = x0.clone()
        if 0 in record_steps:
            saved[0] = x.clone()
        for step in range(1, int(t.max()) + 1):
            x_next = self.q_step(x, step, generator=generator)
            # Картинки, у которых t меньше текущего шага, уже готовы и дальше не меняются.
            active = (step <= t).view(-1, *([1] * (x0.ndim - 1)))
            x = torch.where(active, x_next, x)
            if step in record_steps:
                saved[step] = x.clone()
        return (x, saved) if record is not None else x

    def sample_timesteps(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Случайные шаги t, равномерно из {1, ..., T}, — по одному на объект батча при обучении."""
        return torch.randint(1, self.num_steps + 1, (batch_size,), device=device, generator=generator)

    def snr(self, t: int | torch.Tensor) -> torch.Tensor:
        """Отношение сигнал/шум на шаге t: SNR(t) = ᾱ_t / (1 − ᾱ_t). При t = 0 равно бесконечности.

        Значения заранее посчитаны в float64: при малых t разность 1 − ᾱ_t в float32 теряет точность.
        """
        t = torch.as_tensor(t, device=self.snr_values.device, dtype=torch.long)
        return self.snr_values[t]

    # ------------------------------------------------------------------ #
    # Обратный процесс (роль 2) — добавляется ниже
    # ------------------------------------------------------------------ #
