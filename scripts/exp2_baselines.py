r"""Эксперимент 2, дополнение: сравнение сети с простыми методами шумоподавления.

PSNR восстановления сам по себе не говорит, хорошо ли работает сеть. Поэтому на тех же картинках
и с тем же шумом, что в эксперименте 2, восстановление сетью за один шаг сравнивается с двумя
способами, которые ничего не знают о том, как устроены картинки:

  * обращение масштаба: x_t / sqrt(ᾱ_t) — сигнал возвращается к исходной амплитуде, шум остаётся;
  * то же плюс размытие по Гауссу. Шум мелкозернистый, и размытие убирает его вместе с деталями.
    Сила размытия σ подбирается для каждого t отдельно — по наибольшему PSNR на других картинках
    (из val), чтобы сравнение было честным.

Выигрыш сети над размытием считается по парам «та же картинка — тот же шум»: разность PSNR для
каждой картинки, затем среднее и 95%-й доверительный интервал.

Результаты (пути из configs/default.yaml):
  results/exp2_baselines.csv          — PSNR и SSIM всех способов по шагам t, выбранная σ
  results/exp2_baselines_summary.json — выигрыш сети над размытием по шагам t
  figures/exp2_baselines.png          — PSNR способов и выигрыш сети по шагам t
  figures/exp2_baselines_grid.png     — примеры: x_t, лучшее размытие, сеть

Запуск из корня репозитория (несколько минут; видеокарта не обязательна, нужны веса):
    python -m scripts.exp2_baselines
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from scripts.exp2_reconstruction import autocast, fixed_noise, pick_distinct_classes, write_csv
from src.data import CIFAR10_CLASSES_RU, fixed_examples, load_cifar10_from_config, to_unit_range
from src.diffusion import GaussianDiffusion
from src.metrics import mean_ci, psnr, ssim
from src.plotting import COLORS, REFERENCE, apply_style, save_figure, show_image_grid
from src.training import amp_dtype_for, load_ema_model
from src.unet import UNet
from src.utils import ensure_dir, get_device, get_logger, load_config, resolve_path, save_json, set_seed

METHODS = {
    "rescale": r"деление на $\sqrt{\bar\alpha_t}$",
    "blur": "то же + лучшее размытие",
    "network": "сеть, за 1 шаг",
}
STYLE = {
    "rescale": dict(color=REFERENCE, linestyle="--", marker="o"),
    "blur": dict(color=COLORS["aqua"], linestyle="-", marker="^"),
    "network": dict(color=COLORS["blue"], linestyle="-", marker="o"),
    "full": dict(color=COLORS["orange"], linestyle=":", marker="s"),
}


def rescale(diffusion: GaussianDiffusion, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """x_t / sqrt(ᾱ_t), обрезанное до [−1, 1]: исходная амплитуда сигнала, шум не убран."""
    return (x_t / diffusion.sqrt_alphas_cumprod[t]).clamp(-1.0, 1.0)


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Размытие по Гауссу со стандартным отклонением ``sigma`` пикселей (ядро ±3σ)."""
    return TF.gaussian_blur(x, kernel_size=2 * int(np.ceil(3 * sigma)) + 1, sigma=sigma)


def tune_sigma(diffusion, x0, t, sigmas, seed) -> float:
    """σ размытия с наибольшим средним PSNR на картинках для подбора (не тех, на которых идёт сравнение)."""
    x_t = diffusion.q_sample(x0, t, noise=fixed_noise(x0.shape, seed + 6000 + t, x0.device))
    base, ref = rescale(diffusion, x_t, t), to_unit_range(x0)
    scores = [psnr(to_unit_range(blur(base, s)), ref).mean().item() for s in sigmas]
    return float(sigmas[int(np.argmax(scores))])


@torch.no_grad()
def compare(diffusion, model, x0, x_tune, timesteps, sigmas, batch_size, seed, amp_dtype, keep, logger):
    """Для каждого t: PSNR и SSIM трёх способов, σ размытия, примеры для картинок ``keep``."""
    ref = to_unit_range(x0)
    metrics, examples, chosen = {}, {}, {}
    for t in timesteps:
        sigma = tune_sigma(diffusion, x_tune, t, sigmas, seed)
        # Тот же шум, что в эксперименте 2 (scripts/exp2_reconstruction.py, часть Б).
        x_t = diffusion.q_sample(x0, t, noise=fixed_noise(x0.shape, seed + 1000 + t, x0.device))
        network = []
        for i in range(0, x0.shape[0], batch_size):
            with autocast(x0.device, amp_dtype):
                eps = model(x_t[i:i + batch_size], t)
            network.append(diffusion.predict_x0(x_t[i:i + batch_size], t, eps.float()).clamp(-1, 1))
        base = rescale(diffusion, x_t, t)
        images = {"rescale": base, "blur": blur(base, sigma), "network": torch.cat(network)}
        metrics[t] = {name: (psnr(to_unit_range(img), ref).cpu(), ssim(to_unit_range(img), ref).cpu())
                      for name, img in images.items()}
        examples[t] = {"noisy": x_t[keep].cpu(), **{name: img[keep].cpu() for name, img in images.items()}}
        chosen[t] = sigma
        gain = metrics[t]["network"][0] - metrics[t]["blur"][0]
        logger.info("  t=%4d: σ=%.2f | PSNR: деление %.2f, размытие %.2f, сеть %.2f дБ | выигрыш сети %.2f дБ",
                    t, sigma, metrics[t]["rescale"][0].mean(), metrics[t]["blur"][0].mean(),
                    metrics[t]["network"][0].mean(), gain.mean())
    return metrics, examples, chosen


def read_full_process(path: Path) -> dict[int, float]:
    """PSNR полного обратного процесса из результатов эксперимента 2 (для того же графика), если они есть."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(r["t"]): float(r["psnr"]) for r in csv.DictReader(f) if r["method"] == "full"}


def plot_comparison(rows, gains, full_psnr, path: Path) -> None:
    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in METHODS.items():
        sel = [r for r in rows if r["method"] == name]
        t = np.array([r["t"] for r in sel])
        y, ci = np.array([r["psnr"] for r in sel]), np.array([r["psnr_ci"] for r in sel])
        ax_p.fill_between(t, y - ci, y + ci, color=STYLE[name]["color"], alpha=0.15, linewidth=0)
        ax_p.plot(t, y, label=label, markersize=4, **STYLE[name])
    if full_psnr:
        t_full = sorted(full_psnr)
        ax_p.plot(t_full, [full_psnr[t] for t in t_full], label="сеть, полный процесс (эксп. 2)", markersize=4,
                  **STYLE["full"])
    ax_p.set(title="(а) PSNR относительно исходной картинки", xlabel="Шаг зашумления t", ylabel="PSNR, дБ",
             xlim=(0, 1020))
    ax_p.legend(loc="upper right")

    t = np.array([g["t"] for g in gains])
    y, ci = np.array([g["gain_db"] for g in gains]), np.array([g["gain_ci"] for g in gains])
    ax_g.bar(t, y, width=40, color=COLORS["blue"], alpha=0.85)
    ax_g.errorbar(t, y, yerr=ci, fmt="none", ecolor="#3d3d3a", capsize=3, linewidth=1)
    ax_g.axhline(0, color=REFERENCE, linewidth=1)
    for ti, yi in zip(t, y):
        ax_g.annotate(f"{yi:.1f}", (ti, max(yi, 0)), xytext=(0, 4), textcoords="offset points", ha="center",
                      fontsize=9)
    ax_g.set(title="(б) Выигрыш сети над лучшим размытием", xlabel="Шаг зашумления t",
             ylabel="Разность PSNR, дБ (95% ДИ)", xlim=(0, 1050))
    fig.suptitle("Сеть против простых методов шумоподавления на тех же картинках и шуме", fontweight="bold", y=1.03)
    save_figure(fig, path)


def plot_examples(x0_keep, labels_keep, examples, chosen, grid_timesteps, path: Path) -> None:
    columns, col_labels = [x0_keep], ["оригинал"]
    for t in grid_timesteps:
        for name, short in [("noisy", "$x_t$"), ("blur", f"размытие\nσ = {chosen[t]:g}"), ("network", "сеть")]:
            columns.append(examples[t][name])
            col_labels.append(f"t = {t}\n{short}")
    fig = show_image_grid(torch.stack(columns, dim=1), row_labels=[CIFAR10_CLASSES_RU[int(c)] for c in labels_keep],
                          col_labels=col_labels, title="Лучшее размытие и сеть за один шаг из одной и той же $x_t$",
                          cell_size=0.9)
    save_figure(fig, path)


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
    logger = get_logger("exp2_baselines", log_file=resolve_path(paths["logs_dir"]) / "exp2_baselines.log")
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

    splits = load_cifar10_from_config(cfg)
    # Те же тестовые картинки, что в эксперименте 2 (часть Б), и отдельные картинки из val для подбора σ.
    x0, labels = fixed_examples(splits.test, n=ecfg["n_images"], seed=seed + 5, one_per_class=False)
    x_tune, _ = fixed_examples(splits.val, n=ecfg["blur_tune_images"], seed=seed + 6, one_per_class=False)
    x0, x_tune = x0.to(device), x_tune.to(device)
    keep = pick_distinct_classes(labels, ecfg["grid_images"])
    missing = set(ecfg["baseline_grid_timesteps"]) - set(ecfg["timesteps"])
    if missing:
        raise SystemExit(f"baseline_grid_timesteps должны входить в timesteps, лишние: {sorted(missing)}")

    logger.info("Сравнение на %d тестовых картинках, σ подбирается на %d картинках из val; устройство: %s",
                x0.shape[0], x_tune.shape[0], device)
    metrics, examples, chosen = compare(diffusion, model, x0, x_tune, ecfg["timesteps"], ecfg["blur_sigmas"],
                                        ecfg["batch_size"], seed, amp_dtype, keep, logger)

    rows, gains = [], []
    for t, by_method in metrics.items():
        for name, (p, s) in by_method.items():
            p_mean, p_ci = mean_ci(p)
            s_mean, s_ci = mean_ci(s)
            rows.append({"t": t, "method": name, "blur_sigma": chosen[t] if name == "blur" else "",
                         "psnr": p_mean, "psnr_ci": p_ci, "ssim": s_mean, "ssim_ci": s_ci, "n": p.numel()})
        gain, gain_ci = mean_ci(by_method["network"][0] - by_method["blur"][0])
        gains.append({"t": t, "blur_sigma": chosen[t], "gain_db": gain, "gain_ci": gain_ci,
                      "mse_ratio": 10 ** (gain / 10)})
    write_csv(rows, results_dir / "exp2_baselines.csv")
    save_json({"n_images": int(x0.shape[0]), "tune_images": int(x_tune.shape[0]),
               "gain_over_best_blur": {str(g["t"]): {"blur_sigma": g["blur_sigma"], "gain_db": round(g["gain_db"], 3),
                                                     "gain_ci": round(g["gain_ci"], 3),
                                                     "mse_ratio": round(g["mse_ratio"], 2)} for g in gains}},
              results_dir / "exp2_baselines_summary.json")
    plot_comparison(rows, gains, read_full_process(results_dir / "exp2_reconstruction.csv"),
                    figures_dir / "exp2_baselines.png")
    plot_examples(x0[keep].cpu(), labels[keep], examples, chosen, ecfg["baseline_grid_timesteps"],
                  figures_dir / "exp2_baselines_grid.png")
    logger.info("Готово. Рисунки: %s, таблицы: %s", figures_dir, results_dir)


if __name__ == "__main__":
    main()
