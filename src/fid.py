"""Оценка качества генерации: FID и поиск ближайших соседей в обучающей выборке.

FID (Fréchet Inception Distance, Heusel et al., 2017). Каждая картинка пропускается через сеть
Inception-v3, обученную на ImageNet, и описывается вектором из 2048 признаков. Множество картинок
приближается многомерным нормальным распределением со средним μ и ковариацией Σ этих признаков, и
FID — расстояние Фреше между двумя такими распределениями:

    FID = ‖μ_1 − μ_2‖² + tr(Σ_1 + Σ_2 − 2 · (Σ_1 Σ_2)^{1/2}).

Чем меньше FID, тем ближе сгенерированные картинки к настоящим и по качеству, и по разнообразию.
Используются веса Inception из пакета ``pytorch-fid`` — те же, что в стандартной реализации,
с которой сравнивают результаты в статьях. Веса (~90 МБ) скачиваются при первом вызове.
Сама формула повторяет ``pytorch_fid.fid_score.calculate_frechet_distance``, но работает и с новыми
версиями SciPy, где у ``sqrtm`` убран аргумент ``disp``.

Картинки во всех функциях — массивы numpy uint8 формы [N, 32, 32, 3], как сам CIFAR-10.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import linalg


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """Картинки [N, 3, H, W] в [−1, 1] → массив uint8 [N, H, W, 3], как хранятся настоящие картинки."""
    x = ((images.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def from_uint8(images: np.ndarray) -> torch.Tensor:
    """Массив uint8 [N, H, W, 3] → тензор [N, 3, H, W] в [−1, 1] (для рисования сеток)."""
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def load_inception(device: str | torch.device) -> torch.nn.Module:
    """Сеть Inception-v3 из ``pytorch-fid``, выдающая 2048 признаков (слой перед классификатором).

    Веса скачиваются в кэш torch при первом вызове. Если загрузка однажды оборвалась, в кэше остаётся
    неполный файл, и чтение падает с ошибкой — тогда сообщается, какой файл удалить.
    """
    from pytorch_fid.inception import FID_WEIGHTS_URL, InceptionV3

    try:
        net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    except (RuntimeError, EOFError) as err:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / FID_WEIGHTS_URL.rsplit("/", 1)[-1]
        raise RuntimeError(f"Не удалось прочитать веса Inception-v3: {err}\n"
                           f"Скорее всего, файл {cached} скачался не полностью. "
                           "Удалите его и запустите скрипт снова — веса скачаются заново.") from err
    return net.to(device).eval()


@torch.no_grad()
def inception_features(net: torch.nn.Module, images: np.ndarray, batch_size: int = 100) -> np.ndarray:
    """Признаки Inception для каждой картинки: [N, 2048], float64.

    Сеть сама увеличивает картинки до 299×299 и переводит яркости из [0, 1] в [−1, 1].
    Считается в полной точности (float32), как в стандартной реализации FID.
    """
    device = next(net.parameters()).device
    feats = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[i:i + batch_size])).permute(0, 3, 1, 2)
        x = x.to(device).float().div(255.0)
        feats.append(net(x)[0].flatten(1).double().cpu())
    return torch.cat(feats).numpy()


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray,
                     eps: float = 1e-6) -> float:
    """Расстояние Фреше между N(μ1, Σ1) и N(μ2, Σ2): ‖μ1 − μ2‖² + tr(Σ1 + Σ2 − 2 (Σ1 Σ2)^{1/2}).

    Если квадратный корень из Σ1 Σ2 численно неустойчив, к диагоналям добавляется ``eps``;
    мнимая часть корня, возникающая из-за ошибок округления, отбрасывается.
    """
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Корень из Σ1 Σ2 имеет заметную мнимую часть: {np.abs(covmean.imag).max():.3g}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def fid_from_features(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """FID между двумя наборами признаков Inception."""
    mu_a, sigma_a = feats_a.mean(axis=0), np.cov(feats_a, rowvar=False)
    mu_b, sigma_b = feats_b.mean(axis=0), np.cov(feats_b, rowvar=False)
    return frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


@torch.no_grad()
def nearest_neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    k: int = 1,
    device: str | torch.device = "cpu",
    query_chunk: int = 1024,
    reference_chunk: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой картинки ``query`` — ``k`` ближайших картинок из ``reference`` попиксельно.

    Расстояние — корень из средней по пикселям квадратичной разности яркостей в [0, 1]
    (RMS-расстояние): 0 — одинаковые картинки, 1 — максимально разные. Кандидаты ищутся через
    ‖a − b‖² = ‖a‖² + ‖b‖² − 2⟨a, b⟩ кусками, чтобы не держать в памяти всю матрицу, а расстояния
    до найденных ``k`` соседей затем пересчитываются точно, в float64.

    Returns:
        ``(distances, indices)``, обе формы [N_query, k], отсортированы по возрастанию расстояния.
    """
    q_all = torch.from_numpy(np.ascontiguousarray(query)).reshape(len(query), -1).float().div_(255.0)
    r_all = torch.from_numpy(np.ascontiguousarray(reference)).reshape(len(reference), -1).float().div_(255.0)
    k = min(k, len(reference))
    r_norms = r_all.pow(2).sum(dim=1)

    out_d, out_i = [], []
    for qs in range(0, len(q_all), query_chunk):
        q = q_all[qs:qs + query_chunk].to(device)
        q_norms = q.pow(2).sum(dim=1, keepdim=True)
        best_d = best_i = None
        for rs in range(0, len(r_all), reference_chunk):
            r = r_all[rs:rs + reference_chunk].to(device)
            d2 = q_norms + r_norms[rs:rs + reference_chunk].to(device)[None, :] - 2.0 * q @ r.T
            d, i = d2.topk(min(k, d2.shape[1]), dim=1, largest=False)
            i = i + rs
            if best_d is None:
                best_d, best_i = d, i
            else:
                cand_d, cand_i = torch.cat([best_d, d], dim=1), torch.cat([best_i, i], dim=1)
                best_d, order = cand_d.topk(k, dim=1, largest=False)
                best_i = cand_i.gather(1, order)
        # Точный пересчёт для k кандидатов: формула через скалярные произведения во float32
        # даёт ошибку округления ~1e-4, заметную для почти совпадающих картинок.
        best_i = best_i.cpu()
        q64 = q_all[qs:qs + query_chunk].double()
        exact = torch.stack([(q64 - r_all[best_i[:, j]].double()).pow(2).mean(dim=1) for j in range(k)], dim=1)
        exact, order = exact.sqrt().sort(dim=1)
        out_d.append(exact)
        out_i.append(best_i.gather(1, order))
    return torch.cat(out_d).numpy(), torch.cat(out_i).numpy()
