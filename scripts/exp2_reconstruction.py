r"""Эксперимент 2. Восстановление картинок после зашумления до шага t.

Обученная модель (EMA-веса) проверяется на тестовых картинках CIFAR-10. Три части:

  А. Ошибка предсказания шума по шагам t: MSE(ε, ε_θ(x_t, t)) и соответствующая ей ошибка
     оценки исходной картинки x̂_0 = (x_t − sqrt(1 − ᾱ_t)·ε_θ) / sqrt(ᾱ_t). Показывает, на каких
     шагах сети труднее всего.
  Б. Восстановление: x_0 → x_t по замкнутой формуле → обратно двумя способами:
       * за один шаг — оценка x̂_0 по одному вызову сети;
       * полным обратным процессом — t шагов генерации, начиная с той же x_t.
     Для каждого t считаются PSNR и SSIM относительно x_0 (среднее и 95%-й доверительный интервал).
     Для сравнения те же метрики считаются для самой зашумлённой x_t — это деградация картинки.
  В. Примеры: сетка «оригинал → x_t → восстановления» и несколько восстановлений одной
     и той же x_t с разным шумом обратного процесса.

Результаты (пути из configs/default.yaml):
  results/exp2_noise_error.csv            — ошибка предсказания шума и x̂_0 по шагам t
  results/exp2_reconstruction.csv         — PSNR и SSIM по шагам t и способам восстановления
  results/exp2_summary.json               — ключевые числа для отчёта
  figures/exp2_noise_error.png            — графики части А
  figures/exp2_psnr_ssim.png              — графики части Б
  figures/exp2_reconstruction_grid.png    — сетка примеров
  figures/exp2_stochastic.png             — разные восстановления одной x_t

Запуск из корня репозитория (~30 мин на RTX 3050 Laptop; нужны веса checkpoints/ema_final.pt):
    python -m scripts.exp2_reconstruction
    python -m scripts.exp2_reconstruction --checkpoint checkpoints/ema_step050000.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import CIFAR10_CLASSES_RU, fixed_examples, load_cifar10_from_config, to_unit_range
from src.diffusion import GaussianDiffusion
from src.metrics import mean_ci, psnr, ssim
from src.plotting import COLORS, REFERENCE, apply_style, save_figure, show_image_grid
from src.training import amp_dtype_for, load_ema_model
from src.unet import UNet
from src.utils import ensure_dir, get_device, get_logger, load_config, make_generator, resolve_path, save_json, set_seed

METHODS = {
    "noisy": "зашумлённая $x_t$",
    "one_step": "восстановление за 1 шаг",
    "full": "полный обратный процесс",
}
METHOD_STYLE = {
    "noisy": dict(color=REFERENCE, linestyle="--", marker="o"),
    "one_step": dict(color=COLORS["blue"], linestyle="-", marker="o"),
    "full": dict(color=COLORS["orange"], linestyle="-", marker="s"),
}


def autocast(device: torch.device, dtype: torch.dtype | None):
    """Контекст смешанной точности; при ``dtype=None`` ничего не меняет."""
    return torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None)


def fixed_noise(shape: torch.Size, seed: int, device: torch.device) -> torch.Tensor:
    """Шум из генератора на CPU: одинаковый при запуске на CPU и на видеокарте."""
    return torch.randn(shape, generator=make_generator(seed)).to(device)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pick_distinct_classes(labels: torch.Tensor, n: int) -> list[int]:
    """Индексы первых ``n`` картинок с разными классами — для сеток примеров."""
    picked, seen = [], set()
    for i, c in enumerate(labels.tolist()):
        if c not in seen:
            picked.append(i)
            seen.add(c)
        if len(picked) == n:
            break
    return picked


# --------------------------------------------------------------------------- #
# Часть А. Ошибка предсказания шума
# --------------------------------------------------------------------------- #
@torch.no_grad()
def noise_error(diffusion, model, x0, timesteps, batch_size, seed, amp_dtype) -> list[dict]:
    """Для каждого t: средняя по картинкам MSE предсказанного шума и MSE оценки x̂_0 (без обрезки).

    Ошибки связаны точно: MSE(x̂_0) = (1 − ᾱ_t) / ᾱ_t · MSE(ε) — при больших t маленькая ошибка
    в шуме превращается в большую ошибку в картинке.
    """
    rows = []
    for t in timesteps:
        noise = fixed_noise(x0.shape, seed + t, x0.device)
        err_eps, err_x0 = [], []
        for i in range(0, x0.shape[0], batch_size):
            x0_b, noise_b = x0[i:i + batch_size], noise[i:i + batch_size]
            x_t = diffusion.q_sample(x0_b, t, noise=noise_b)
            with autocast(x0.device, amp_dtype):
                eps = model(x_t, t)
            eps = eps.float()
            x0_pred = diffusion.predict_x0(x_t, t, eps)
            err_eps.append((eps - noise_b).pow(2).flatten(1).mean(1))
            err_x0.append((x0_pred - x0_b).pow(2).flatten(1).mean(1))
        eps_mean, eps_ci = mean_ci(torch.cat(err_eps))
        x0_mean, x0_ci = mean_ci(torch.cat(err_x0))
        rows.append({"t": t, "alpha_bar": diffusion.alphas_cumprod[t].item(),
                     "mse_eps": eps_mean, "mse_eps_ci": eps_ci, "mse_x0": x0_mean, "mse_x0_ci": x0_ci})
    return rows


def plot_noise_error(rows: list[dict], path: Path) -> None:
    t = np.array([r["t"] for r in rows])
    fig, (ax_e, ax_x) = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, key, title, ylabel in [
        (ax_e, "mse_eps", "(а) Ошибка предсказания шума", r"MSE$(\varepsilon,\ \varepsilon_\theta)$"),
        (ax_x, "mse_x0", r"(б) Ошибка оценки картинки $\hat x_0$ за 1 шаг", r"MSE$(x_0,\ \hat x_0)$"),
    ]:
        y = np.array([r[key] for r in rows])
        ci = np.array([r[key + "_ci"] for r in rows])
        ax.fill_between(t, np.maximum(y - ci, 1e-12), y + ci, color=COLORS["blue"], alpha=0.18, linewidth=0)
        ax.plot(t, y, "o-", color=COLORS["blue"], markersize=4)
        ax.set_yscale("log")
        ax.set(title=title, xlabel="Шаг t", ylabel=ylabel + ", логарифмическая шкала", xlim=(0, 1010))
    fig.suptitle("Где сети труднее: ошибка на тестовых картинках по шагам t (95% ДИ — полоса)",
                 fontweight="bold", y=1.03)
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Часть Б. Восстановление
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruction(diffusion, model, x0, timesteps, batch_size, seed, amp_dtype, sampling, keep, logger):
    """PSNR и SSIM для трёх вариантов на каждом шаге t, плюс сохранённые примеры для картинок ``keep``.

    Оба способа восстановления стартуют с одной и той же x_t.
    """
    ref = to_unit_range(x0)
    metrics: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    examples: dict[int, dict[str, torch.Tensor]] = {}
    for t in timesteps:
        started = time.perf_counter()
        x_t_all = diffusion.q_sample(x0, t, noise=fixed_noise(x0.shape, seed + 1000 + t, x0.device))
        gen = make_generator(seed + 2000 + t, x0.device)
        one_step, full = [], []
        for i in range(0, x0.shape[0], batch_size):
            x_t = x_t_all[i:i + batch_size]
            with autocast(x0.device, amp_dtype):
                eps = model(x_t, t)
            one_step.append(diffusion.predict_x0(x_t, t, eps.float()).clamp(-1, 1))
            full.append(diffusion.p_sample_loop(model, x_start=x_t, t_start=t, generator=gen,
                                                variance=sampling["variance"], clip_x0=sampling["clip_x0"],
                                                autocast_dtype=amp_dtype).images)
        images = {"noisy": x_t_all, "one_step": torch.cat(one_step), "full": torch.cat(full)}
        metrics[t] = {}
        for name, img in images.items():
            unit = to_unit_range(img)
            metrics[t][name] = (psnr(unit, ref).cpu(), ssim(unit, ref).cpu())
        examples[t] = {name: img[keep].cpu() for name, img in images.items()}
        logger.info("  t=%4d: PSNR за 1 шаг %.2f дБ, процессом %.2f дБ, x_t %.2f дБ (%.0f с)", t,
                    metrics[t]["one_step"][0].mean(), metrics[t]["full"][0].mean(),
                    metrics[t]["noisy"][0].mean(), time.perf_counter() - started)
    return metrics, examples


def reconstruction_rows(metrics, diffusion) -> list[dict]:
    rows = []
    for t, by_method in metrics.items():
        for name, (p, s) in by_method.items():
            p_mean, p_ci = mean_ci(p)
            s_mean, s_ci = mean_ci(s)
            rows.append({"t": t, "alpha_bar": diffusion.alphas_cumprod[t].item(), "method": name,
                         "psnr": p_mean, "psnr_ci": p_ci, "ssim": s_mean, "ssim_ci": s_ci, "n": p.numel()})
    return rows


def plot_psnr_ssim(rows: list[dict], path: Path) -> None:
    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in METHODS.items():
        sel = [r for r in rows if r["method"] == name]
        t = np.array([r["t"] for r in sel])
        style = METHOD_STYLE[name]
        for ax, key in [(ax_p, "psnr"), (ax_s, "ssim")]:
            y = np.array([r[key] for r in sel])
            ci = np.array([r[key + "_ci"] for r in sel])
            ax.fill_between(t, y - ci, y + ci, color=style["color"], alpha=0.15, linewidth=0)
            ax.plot(t, y, label=label, markersize=4, **style)
    ax_p.set(title="(а) PSNR относительно исходной картинки", xlabel="Шаг зашумления t", ylabel="PSNR, дБ",
             xlim=(0, 1020))
    ax_s.set(title="(б) SSIM относительно исходной картинки", xlabel="Шаг зашумления t", ylabel="SSIM",
             xlim=(0, 1020))
    ax_p.legend(loc="upper right")
    fig.suptitle("Качество восстановления в зависимости от шага зашумления (95% ДИ — полоса)", fontweight="bold",
                 y=1.03)
    save_figure(fig, path)


def plot_examples(x0_keep, labels_keep, examples, grid_timesteps, path: Path) -> None:
    """Сетка: строки — картинки, столбцы — оригинал и для каждого t: x_t, за 1 шаг, процесс."""
    columns = [x0_keep]
    col_labels = ["оригинал"]
    for t in grid_timesteps:
        for name, short in [("noisy", "$x_t$"), ("one_step", "1 шаг"), ("full", "процесс")]:
            columns.append(examples[t][name])
            col_labels.append(f"t = {t}\n{short}")
    grid = torch.stack(columns, dim=1)
    fig = show_image_grid(grid, row_labels=[CIFAR10_CLASSES_RU[int(c)] for c in labels_keep], col_labels=col_labels,
                          title="Зашумление до шага t и восстановление: за 1 шаг и полным обратным процессом",
                          cell_size=0.9)
    save_figure(fig, path)


@torch.no_grad()
def plot_stochastic(diffusion, model, x0, labels, t, n_samples, seed, amp_dtype, sampling, path: Path) -> None:
    """Несколько восстановлений одной и той же x_t: обратный процесс случаен, а восстановление за 1 шаг — нет."""
    device = x0.device
    x_t = diffusion.q_sample(x0, t, noise=fixed_noise(x0.shape, seed + 3000, device))
    with autocast(device, amp_dtype):
        eps = model(x_t, t)
    one_step = diffusion.predict_x0(x_t, t, eps.float()).clamp(-1, 1)
    out = diffusion.p_sample_loop(model, x_start=x_t.repeat_interleave(n_samples, dim=0), t_start=t,
                                  generator=make_generator(seed + 4000, device), variance=sampling["variance"],
                                  clip_x0=sampling["clip_x0"], autocast_dtype=amp_dtype)
    samples = out.images.reshape(x0.shape[0], n_samples, *x0.shape[1:])
    grid = torch.cat([x0[:, None], x_t[:, None], one_step[:, None], samples], dim=1).cpu()
    col_labels = ["оригинал", f"$x_t$, t = {t}", "1 шаг"] + [f"процесс {k + 1}" for k in range(n_samples)]
    fig = show_image_grid(grid, row_labels=[CIFAR10_CLASSES_RU[int(c)] for c in labels], col_labels=col_labels,
                          title=f"Разные восстановления одной и той же зашумлённой картинки (t = {t})", cell_size=0.9)
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="путь к конфигу")
    parser.add_argument("--checkpoint", default=None, help="веса модели вместо experiment2.checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    ecfg = cfg["experiment2"]
    paths = cfg["paths"]
    logger = get_logger("exp2", log_file=resolve_path(paths["logs_dir"]) / "exp2_reconstruction.log")
    results_dir = ensure_dir(paths["results_dir"])
    figures_dir = ensure_dir(paths["figures_dir"])
    apply_style()

    device = get_device()
    amp_dtype = amp_dtype_for(device)
    checkpoint = resolve_path(args.checkpoint or ecfg["checkpoint"])
    if not checkpoint.exists():
        raise SystemExit(f"Нет файла весов {checkpoint}. Скачайте его по ссылке из README (раздел «Веса моделей»).")
    model = load_ema_model(UNet.from_config(cfg), checkpoint, device)
    diffusion = GaussianDiffusion.from_config(cfg).to(device)
    batch_size = ecfg["batch_size"]
    logger.info("Веса: %s; устройство: %s; смешанная точность: %s", checkpoint.name, device, amp_dtype or "нет")

    splits = load_cifar10_from_config(cfg)

    logger.info("Часть А: ошибка предсказания шума на %d картинках", ecfg["error_images"])
    x_err, _ = fixed_examples(splits.test, n=ecfg["error_images"], seed=seed + 4, one_per_class=False)
    error_rows = noise_error(diffusion, model, x_err.to(device), ecfg["error_timesteps"], batch_size, seed, amp_dtype)
    write_csv(error_rows, results_dir / "exp2_noise_error.csv")
    plot_noise_error(error_rows, figures_dir / "exp2_noise_error.png")
    for r in error_rows:
        logger.info("  t=%4d: MSE шума %.4f, MSE x0 %.4f", r["t"], r["mse_eps"], r["mse_x0"])

    logger.info("Часть Б: восстановление %d картинок с шагов %s", ecfg["n_images"], ecfg["timesteps"])
    x0, labels = fixed_examples(splits.test, n=ecfg["n_images"], seed=seed + 5, one_per_class=False)
    x0 = x0.to(device)
    keep = pick_distinct_classes(labels, ecfg["grid_images"])
    missing = set(ecfg["grid_timesteps"]) - set(ecfg["timesteps"])
    if missing:
        raise SystemExit(f"grid_timesteps должны входить в timesteps, лишние: {sorted(missing)}")
    metrics, examples = reconstruction(diffusion, model, x0, ecfg["timesteps"], batch_size, seed, amp_dtype,
                                       cfg["sampling"], keep, logger)
    rows = reconstruction_rows(metrics, diffusion)
    write_csv(rows, results_dir / "exp2_reconstruction.csv")
    plot_psnr_ssim(rows, figures_dir / "exp2_psnr_ssim.png")

    logger.info("Часть В: примеры")
    plot_examples(x0[keep].cpu(), labels[keep], examples, ecfg["grid_timesteps"],
                  figures_dir / "exp2_reconstruction_grid.png")
    k = ecfg["stochastic_images"]
    plot_stochastic(diffusion, model, x0[keep[:k]], labels[keep[:k]], ecfg["stochastic_t"],
                    ecfg["stochastic_samples"], seed, amp_dtype, cfg["sampling"], figures_dir / "exp2_stochastic.png")

    by_t = {str(t): {name: {"psnr": round(r["psnr"], 3), "ssim": round(r["ssim"], 4)}
                     for name in METHODS for r in rows if r["t"] == t and r["method"] == name}
            for t in ecfg["timesteps"]}
    save_json({
        "checkpoint": checkpoint.name,
        "n_images": ecfg["n_images"],
        "reconstruction": by_t,
        "noise_error": {str(r["t"]): {"mse_eps": round(r["mse_eps"], 5), "mse_x0": round(r["mse_x0"], 5)}
                        for r in error_rows},
    }, results_dir / "exp2_summary.json")
    logger.info("Готово. Рисунки: %s, таблицы: %s", figures_dir, results_dir)


if __name__ == "__main__":
    main()
