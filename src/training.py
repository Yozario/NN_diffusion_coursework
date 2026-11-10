"""Части цикла обучения: EMA-копия весов, порядок батчей с продолжением, разогрев шага обучения, чекпойнты.

Сам цикл обучения — в ``scripts/train_diffusion.py``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import Sampler


# --------------------------------------------------------------------------- #
# EMA-копия весов
# --------------------------------------------------------------------------- #
class EMA:
    """Экспоненциальное скользящее среднее весов модели: θ_ema ← d · θ_ema + (1 − d) · θ.

    Веса во время обучения «дрожат» от шага к шагу; их усреднение по последним тысячам шагов
    даёт заметно более чистые картинки при генерации (так сделано в DDPM, d = 0.9999).
    При d = 0.9999 копия помнит примерно последние 1/(1 − d) = 10 000 шагов, поэтому в начале
    обучения d растёт постепенно: d_k = min(d, (1 + k) / (10 + k)), где k — номер обновления.
    Иначе первые десятки тысяч шагов EMA-копия была бы почти случайной инициализацией.

    Args:
        model: обучаемая модель; копия создаётся сразу и дальше хранится отдельно.
        decay: целевой коэффициент d.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay должен лежать в (0, 1), получено {decay}")
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    def current_decay(self) -> float:
        """Коэффициент d для следующего обновления."""
        k = self.num_updates + 1
        return min(self.decay, (1 + k) / (10 + k))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Сдвигает EMA-веса к текущим весам модели (вызывать после каждого шага оптимизатора)."""
        d = self.current_decay()
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - d)          # ema = ema + (1 − d)(p − ema) = d·ema + (1 − d)·p
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "model": self.model.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.model.load_state_dict(state["model"])


# --------------------------------------------------------------------------- #
# Порядок батчей
# --------------------------------------------------------------------------- #
class EpochBatchSampler(Sampler[list[int]]):
    """Батчи индексов, перемешанные заново в каждой эпохе, с возможностью начать с середины эпохи.

    Порядок в эпохе зависит только от ``seed`` и номера эпохи, поэтому после остановки обучения
    можно продолжить ровно с того батча, на котором остановились: ``set_position(эпоха, батч)``.
    Неполный последний батч отбрасывается, чтобы все батчи были одного размера.
    """

    def __init__(self, num_samples: int, batch_size: int, seed: int) -> None:
        if batch_size > num_samples:
            raise ValueError("Размер батча больше числа объектов")
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0

    @property
    def batches_per_epoch(self) -> int:
        return self.num_samples // self.batch_size

    def set_position(self, epoch: int, start_batch: int = 0) -> None:
        """Следующий проход начнётся с батча ``start_batch`` эпохи ``epoch``."""
        if not 0 <= start_batch < self.batches_per_epoch:
            raise ValueError(f"start_batch должен лежать в 0..{self.batches_per_epoch - 1}")
        self.epoch, self.start_batch = epoch, start_batch

    def epoch_permutation(self, epoch: int) -> torch.Tensor:
        """Перестановка индексов для эпохи — одна и та же при каждом вызове."""
        gen = torch.Generator().manual_seed(self.seed * 100_003 + epoch)
        return torch.randperm(self.num_samples, generator=gen)

    def __iter__(self) -> Iterator[list[int]]:
        perm = self.epoch_permutation(self.epoch)
        for b in range(self.start_batch, self.batches_per_epoch):
            yield perm[b * self.batch_size:(b + 1) * self.batch_size].tolist()

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch


# --------------------------------------------------------------------------- #
# Шаг обучения
# --------------------------------------------------------------------------- #
def warmup_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    """Скорость обучения на шаге ``step`` (с нуля): линейный рост за ``warmup_steps`` шагов, затем постоянная.

    Разогрев защищает от слишком больших обновлений в самом начале, когда оценки моментов
    в Adam ещё неточные (в DDPM — 5000 шагов, у нас обучение короче).
    """
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup_steps)


def amp_dtype_for(device: torch.device, enabled: bool = True) -> torch.dtype | None:
    """Тип смешанной точности: bfloat16, если видеокарта его поддерживает, иначе float16; на CPU — None."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# --------------------------------------------------------------------------- #
# Чекпойнты
# --------------------------------------------------------------------------- #
def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Сохраняет чекпойнт атомарно: сначала во временный файл, затем переименование.

    Если обучение прервётся во время записи, старый чекпойнт останется целым.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Загружает чекпойнт. ``weights_only=True``: из файла читаются только тензоры и простые типы."""
    return torch.load(path, map_location=device, weights_only=True)


def load_ema_model(model: nn.Module, path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Загружает в модель EMA-веса из чекпойнта (полного ``last.pt`` или отдельного ``ema_*.pt``)."""
    state = load_checkpoint(path, device)
    weights = state["ema"]["model"] if "ema" in state else state["model"]
    model.load_state_dict(weights)
    return model.to(device).eval()
