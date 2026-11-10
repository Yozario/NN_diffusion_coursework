r"""Гауссовский диффузионный процесс (DDPM, Ho et al., 2020).

Обозначения совпадают со статьёй. x_0 — исходная картинка, x_t — картинка после t шагов
зашумления, t = 0, 1, ..., T. Один шаг прямого процесса:

    x_t = sqrt(1 − β_t) · x_{t−1} + sqrt(β_t) · ε,   ε ~ N(0, I).

Отсюда по индукции следует формула в замкнутом виде — x_t можно получить из x_0 сразу,
без прохода по всем промежуточным шагам:

    x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε,   α_t = 1 − β_t,   ᾱ_t = α_1 · α_2 · … · α_t.

Обратный процесс идёт от чистого шума x_T к картинке x_0. Сеть ε_θ(x_t, t) предсказывает шум,
по нему восстанавливается оценка исходной картинки

    x̂_0 = (x_t − sqrt(1 − ᾱ_t) · ε_θ(x_t, t)) / sqrt(ᾱ_t),

и делается шаг к x_{t−1} по апостериорному распределению q(x_{t−1} | x_t, x_0) с x̂_0 вместо x_0:

    x_{t−1} = c1_t · x̂_0 + c2_t · x_t + σ_t · z,   z ~ N(0, I),
    c1_t = sqrt(ᾱ_{t−1}) · β_t / (1 − ᾱ_t),   c2_t = sqrt(α_t) · (1 − ᾱ_{t−1}) / (1 − ᾱ_t).

Без обрезки x̂_0 это ровно формула из статьи: x_{t−1} = (x_t − β_t / sqrt(1 − ᾱ_t) · ε_θ) / sqrt(α_t) + σ_t · z.
Обрезка x̂_0 до [−1, 1] (как в официальном коде DDPM) не даёт генерации уходить за диапазон данных.

Соглашение об индексах в коде: все массивы коэффициентов имеют длину T + 1, а индекс равен
номеру шага. Нулевой элемент соответствует отсутствию шума: β_0 = 0, ᾱ_0 = 1, поэтому
``q_sample(x0, t=0)`` возвращает саму x_0. При обучении шаг t выбирается из {1, ..., T}.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


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
@dataclass
class SampleOutput:
    """Результат обратного процесса.

    Attributes:
        images: итоговые картинки x_0 [B, C, H, W] в диапазоне примерно [−1, 1], на устройстве модели.
        x_t: ``{шаг: x_t}`` для запрошенных шагов, копии на CPU.
        x0_pred: ``{шаг: x̂_0}`` — оценка исходной картинки, сделанная сетью на этом шаге, копии на CPU.
    """

    images: torch.Tensor
    x_t: dict[int, torch.Tensor] = field(default_factory=dict)
    x0_pred: dict[int, torch.Tensor] = field(default_factory=dict)


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

        # Обратный процесс. Коэффициенты определены для t >= 1; на шаге 0 стоят нули.
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])  # ᾱ_{t−1}
        one_minus_abar = (1.0 - alphas_cumprod).clamp(min=1e-20)                       # без деления на 0 при t=0
        posterior_variance = betas * (1.0 - abar_prev) / one_minus_abar
        register("sqrt_recip_alphas_cumprod", alphas_cumprod.rsqrt())                  # 1 / sqrt(ᾱ_t)
        register("sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1.0).sqrt())   # sqrt(1/ᾱ_t − 1)
        register("posterior_mean_coef1", betas * abar_prev.sqrt() / one_minus_abar)      # c1_t
        register("posterior_mean_coef2", (1.0 - abar_prev) * alphas.sqrt() / one_minus_abar)  # c2_t
        register("posterior_variance", posterior_variance)                               # β̃_t; β̃_1 = 0

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
    # Прямой процесс
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
    # Функция потерь
    # ------------------------------------------------------------------ #
    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Упрощённая функция потерь DDPM: L = ‖ε − ε_θ(x_t, t)‖², среднее по батчу и пикселям.

        Для каждой картинки берётся случайный шаг t из {1, ..., T} и случайный шум ε,
        по замкнутой формуле строится x_t, и сеть учится угадывать добавленный шум.
        Шаг и шум можно передать явно — так считается валидационная потеря на фиксированных
        (t, ε), чтобы её значения на разных этапах обучения были сравнимы.
        Потеря считается во float32, даже если сеть работает в смешанной точности.
        """
        if t is None:
            t = self.sample_timesteps(x0.shape[0], x0.device, generator)
        if noise is None:
            noise = self._randn_like(x0, generator)
        x_t = self.q_sample(x0, t, noise=noise)
        return F.mse_loss(model(x_t, t).float(), noise.float())

    # ------------------------------------------------------------------ #
    # Обратный процесс
    # ------------------------------------------------------------------ #
    def predict_x0(self, x_t: torch.Tensor, t: int | torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Оценка исходной картинки по x_t и шуму: x̂_0 = (x_t − sqrt(1 − ᾱ_t) · ε) / sqrt(ᾱ_t)."""
        t = self._timesteps(t, x_t.shape[0], x_t.device)
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.ndim) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.ndim) * eps)

    def q_posterior_mean(self, x0: torch.Tensor, x_t: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Среднее апостериорного распределения q(x_{t−1} | x_t, x_0) = c1_t · x_0 + c2_t · x_t."""
        t = self._timesteps(t, x_t.shape[0], x_t.device)
        return (self._extract(self.posterior_mean_coef1, t, x_t.ndim) * x0
                + self._extract(self.posterior_mean_coef2, t, x_t.ndim) * x_t)

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        generator: torch.Generator | None = None,
        clip_x0: bool = True,
        variance: str = "beta",
        autocast_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Один шаг обратного процесса x_t → x_{t−1}, шаг t общий для всего батча.

        Args:
            model: сеть ε_θ(x_t, t).
            x_t: текущие картинки [B, C, H, W], float32.
            t: номер шага, 1..T.
            generator: генератор шума z (на том же устройстве, что x_t).
            clip_x0: обрезать оценку x̂_0 до [−1, 1].
            variance: дисперсия шума σ_t²: ``"beta"`` — β_t (как в DDPM для CIFAR-10),
                ``"posterior"`` — β̃_t = β_t · (1 − ᾱ_{t−1}) / (1 − ᾱ_t).
            autocast_dtype: тип смешанной точности для прохода сети (например, ``torch.bfloat16``);
                ``None`` — без неё. Сам шаг процесса всегда считается во float32.

        Returns:
            ``(x_{t−1}, x̂_0)``. На последнем шаге (t = 1) шум не добавляется.
        """
        if not 1 <= t <= self.num_steps:
            raise ValueError(f"Шаг обратного процесса должен лежать в 1..{self.num_steps}, получено {t}")
        if variance not in ("beta", "posterior"):
            raise ValueError(f"variance должна быть 'beta' или 'posterior', получено {variance!r}")

        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        amp = torch.autocast(x_t.device.type, dtype=autocast_dtype) if autocast_dtype else nullcontext()
        with amp:
            eps = model(x_t, t_batch)
        x0_pred = self.predict_x0(x_t, t_batch, eps.float())
        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        mean = self.q_posterior_mean(x0_pred, x_t, t_batch)
        if t == 1:
            return mean, x0_pred
        var = self.betas[t] if variance == "beta" else self.posterior_variance[t]
        return mean + var.sqrt() * self._randn_like(x_t, generator), x0_pred

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...] | None = None,
        x_start: torch.Tensor | None = None,
        t_start: int | None = None,
        generator: torch.Generator | None = None,
        record_steps: Iterable[int] | None = None,
        clip_x0: bool = True,
        variance: str = "beta",
        autocast_dtype: torch.dtype | None = None,
        progress: bool = False,
    ) -> "SampleOutput":
        """Обратный процесс от шага ``t_start`` до 0.

        Два режима:
            * генерация с нуля — задать ``shape``: x_T ~ N(0, I), ``t_start`` = T;
            * восстановление — задать ``x_start`` (картинки на шаге ``t_start``, например из
              :meth:`q_sample`) и ``t_start``: процесс пройдёт только оставшиеся шаги.

        Сеть на время генерации переводится в режим ``eval`` (без dropout), затем режим возвращается.

        Args:
            model: сеть ε_θ(x_t, t).
            shape: форма батча для генерации с нуля, например ``(64, 3, 32, 32)``.
            x_start: начальные картинки для восстановления.
            t_start: начальный шаг; по умолчанию T.
            generator: генератор шума на устройстве модели — для воспроизводимости.
            record_steps: шаги, на которых сохранить x_t и x̂_0 (копии на CPU), например
                ``range(0, 1001, 10)``. Нужны для анализа траекторий.
            clip_x0, variance, autocast_dtype: см. :meth:`p_sample`.
            progress: показывать полосу прогресса.

        Returns:
            :class:`SampleOutput` с итоговыми картинками и записанной траекторией.
        """
        t_start = self.num_steps if t_start is None else int(t_start)
        if not 0 <= t_start <= self.num_steps:
            raise ValueError(f"t_start должен лежать в 0..{self.num_steps}")
        device = next(model.parameters()).device
        if x_start is not None:
            x = x_start.to(device=device, dtype=torch.float32).clone()
        elif shape is not None:
            x = torch.randn(shape, generator=generator, device=device)
        else:
            raise ValueError("Нужно задать shape (генерация с нуля) или x_start (восстановление)")

        record = set(record_steps or [])
        out = SampleOutput(images=x)
        if t_start in record:
            out.x_t[t_start] = x.cpu()

        was_training = model.training
        model.eval()
        try:
            for t in tqdm(range(t_start, 0, -1), disable=not progress, desc="Генерация", leave=False):
                x, x0_pred = self.p_sample(model, x, t, generator=generator, clip_x0=clip_x0,
                                           variance=variance, autocast_dtype=autocast_dtype)
                if t in record:
                    out.x0_pred[t] = x0_pred.cpu()
                if t - 1 in record:
                    out.x_t[t - 1] = x.cpu()
        finally:
            model.train(was_training)
        out.images = x
        return out

    @torch.no_grad()
    def reconstruct(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: int,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> "SampleOutput":
        """Зашумляет картинки до шага t прямым процессом и восстанавливает обратным.

        Показывает, сколько информации об исходной картинке сеть способна вернуть
        после t шагов зашумления. Остальные аргументы — как у :meth:`p_sample_loop`.
        """
        x_t = self.q_sample(x0, t, generator=generator)
        return self.p_sample_loop(model, x_start=x_t, t_start=t, generator=generator, **kwargs)
