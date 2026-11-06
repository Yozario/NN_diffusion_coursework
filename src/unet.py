r"""U-Net, предсказывающий шум ε по зашумлённой картинке x_t и номеру шага t.

Архитектура следует DDPM (Ho et al., 2020), уменьшенная под видеокарту с 4 ГБ памяти:

    вход x_t [B, 3, 32, 32], шаг t [B]
      │
      ├─ t → синусоидальное кодирование → MLP → вектор шага temb [B, 4·C]
      │
    conv 3×3: 3 → C
      │
    спуск (encoder): на каждом уровне разрешения несколько остаточных блоков,
      затем уменьшение разрешения вдвое. Выходы всех блоков запоминаются
      как «пропускные соединения» (skip connections).
      │   32×32 → 16×16 → 8×8
    середина: остаточный блок → внимание → остаточный блок (8×8)
      │
    подъём (decoder): на каждом уровне блоки получают на вход ещё и
      сохранённый выход симметричного блока спуска (конкатенация по каналам),
      затем увеличение разрешения вдвое.
      │   8×8 → 16×16 → 32×32
    GroupNorm → SiLU → conv 3×3: C → 3   ⇒   предсказанный шум ε̂ [B, 3, 32, 32]

Номер шага подаётся в каждый остаточный блок: модель одна на все t и должна знать,
насколько сильно зашумлён вход. Слои внимания (self-attention) стоят на разрешении 16×16:
там каждый пиксель может «посмотреть» на всю картинку, что помогает согласовать
крупные детали. На 32×32 внимание не ставим — оно квадратично по числу пикселей
и заняло бы слишком много памяти.

Последние свёртки остаточных блоков, внимания и выходного слоя инициализируются нулями.
Тогда в начале обучения каждый блок — тождественное отображение, а сеть целиком
выдаёт ноль; так обучение стартует устойчивее (тот же приём, что в DDPM).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Синусоидальное кодирование номера шага, как позиционное кодирование в трансформере.

    Номер шага t превращается в вектор из ``dim`` чисел: синусы и косинусы t с частотами
    от 1 до 1/``max_period`` в геометрической прогрессии. Близкие t дают близкие векторы,
    далёкие — заметно разные, и сети легко отличать шаги друг от друга.

    Args:
        t: номера шагов [B] (целые или вещественные).
        dim: длина вектора, чётное число.

    Returns:
        Тензор [B, dim].
    """
    if dim % 2 != 0:
        raise ValueError(f"Размер кодирования шага должен быть чётным, получено {dim}")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm с 32 группами (или меньше, если каналов мало).

    Нормирует признаки внутри групп каналов каждой картинки отдельно — в отличие
    от BatchNorm не зависит от размера батча, что важно при маленьких батчах.
    """
    return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=1e-6)


def zero_init(module: nn.Module) -> nn.Module:
    """Обнуляет веса и смещения слоя и возвращает его."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# --------------------------------------------------------------------------- #
# Блоки
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    """Остаточный блок с учётом номера шага.

    h = conv(SiLU(GN(x)));  h = h + Linear(temb);  h = conv(dropout(SiLU(GN(h))));  выход = skip(x) + h,
    где skip — тождество или свёртка 1×1, если число каналов меняется.
    """

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]  # шаг добавляется к каждому пикселю
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Многоголовое самовнимание по всем пикселям карты признаков, с остаточной связью.

    Каждый пиксель — «токен» длины C. Для разрешения 16×16 это 256 токенов.
    """

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"Число каналов {channels} должно делиться на число голов {num_heads}")
        self.num_heads = num_heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = zero_init(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))                                   # [B, 3C, H, W]
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = (qkv[:, i].transpose(-1, -2) for i in range(3))      # каждый [B, heads, HW, C/heads]
        out = F.scaled_dot_product_attention(q, k, v)                  # softmax(QKᵀ/√d)·V
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Уменьшение разрешения вдвое свёрткой 3×3 с шагом 2."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Увеличение разрешения вдвое: повтор ближайшего пикселя, затем свёртка 3×3."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #
class UNet(nn.Module):
    """U-Net для предсказания шума: ``model(x_t, t) → ε̂``.

    Args:
        image_size: сторона входной картинки (32 для CIFAR-10).
        in_channels: каналов во входе и выходе (3 для RGB).
        base_channels: C — число каналов на первом уровне.
        channel_mults: множители каналов по уровням; длина задаёт число уровней.
            Например, (1, 2, 2): 32×32 — C, 16×16 — 2C, 8×8 — 2C.
        num_res_blocks: остаточных блоков на каждом уровне спуска (на подъёме на один больше).
        attn_resolutions: разрешения, на которых после блоков стоит внимание.
        num_heads: число голов внимания.
        dropout: доля зануляемых признаков внутри остаточных блоков (против переобучения).
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % 2 ** (len(channel_mults) - 1) != 0:
            raise ValueError(f"Сторону {image_size} нельзя {len(channel_mults) - 1} раз поделить пополам")
        self.base_channels = base_channels
        temb_dim = 4 * base_channels

        # Кодирование номера шага: синусы/косинусы → двухслойный перцептрон.
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Спуск ----
        self.down = nn.ModuleList()
        skip_channels = [base_channels]   # каналы всех сохраняемых выходов, по порядку
        ch, res = base_channels, image_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = nn.ModuleList([ResBlock(ch, out_ch, temb_dim, dropout)])
                if res in attn_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads))
                self.down.append(layers)
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                res //= 2
                skip_channels.append(ch)

        # ---- Середина ----
        self.mid_block1 = ResBlock(ch, ch, temb_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_block2 = ResBlock(ch, ch, temb_dim, dropout)

        # ---- Подъём ----
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                layers = nn.ModuleList([ResBlock(ch + skip_channels.pop(), out_ch, temb_dim, dropout)])
                if res in attn_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads))
                self.up.append(layers)
                ch = out_ch
            if level != 0:
                self.up.append(nn.ModuleList([Upsample(ch)]))
                res *= 2
        assert not skip_channels, "каждому выходу спуска должен соответствовать блок подъёма"

        self.norm_out = group_norm(ch)
        self.conv_out = zero_init(nn.Conv2d(ch, in_channels, 3, padding=1))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "UNet":
        """Создаёт сеть по разделу ``model`` конфига проекта."""
        m = cfg["model"]
        return cls(
            image_size=m["image_size"],
            in_channels=m["in_channels"],
            base_channels=m["base_channels"],
            channel_mults=tuple(m["channel_mults"]),
            num_res_blocks=m["num_res_blocks"],
            attn_resolutions=tuple(m["attn_resolutions"]),
            num_heads=m["num_heads"],
            dropout=m["dropout"],
        )

    def forward(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Предсказывает шум.

        Args:
            x: зашумлённые картинки x_t [B, 3, H, W].
            t: номера шагов — число для всего батча или тензор [B].

        Returns:
            Предсказанный шум ε̂ той же формы, что x.
        """
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))

        h = self.conv_in(x)
        skips = [h]
        for layers in self.down:
            for layer in layers:
                h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for layers in self.up:
            for layer in layers:
                if isinstance(layer, ResBlock):
                    h = layer(torch.cat([h, skips.pop()], dim=1), temb)
                else:
                    h = layer(h)

        return self.conv_out(F.silu(self.norm_out(h)))


def count_parameters(model: nn.Module) -> int:
    """Число обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
