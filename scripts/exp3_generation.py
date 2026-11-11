r"""Эксперимент 3. Генерация картинок из чистого шума и оценка их качества.

Обученная модель (EMA-веса) генерирует картинки полным обратным процессом: x_T ~ N(0, I) → x_0
за T = 1000 шагов. Четыре части:

  А. Генерация N картинок. После каждого батча картинки сохраняются в results/raw/: если запуск
     прервать, повторный продолжит с того же батча и даст те же самые картинки.
  Б. FID — расстояние между распределениями признаков Inception-v3 у сгенерированных картинок
     и у 50 000 обучающих картинок CIFAR-10. FID зависит от числа картинок (на малых выборках он
     завышен), поэтому тем же способом считается FID у N настоящих тестовых картинок — это лучшее
     значение, достижимое при таком N.
  В. Проверка на запоминание: для каждой сгенерированной и каждой тестовой картинки — расстояние до
     ближайшей обучающей картинки. Если бы модель копировала обучающие картинки, у сгенерированных
     эти расстояния были бы заметно меньше, чем у тестовых, которых модель не видела.
  Г. Рисунки: сетка сгенерированных картинок, ближайшие обучающие соседи, траектория генерации.

Результаты (пути из configs/default.yaml):
  results/raw/exp3_samples_<variance>.npz — сгенерированные картинки (не хранятся в git)
  results/exp3_summary.json               — FID и статистика расстояний до обучающих картинок
  figures/exp3_samples.png                — сетка сгенерированных картинок
  figures/exp3_nn_distances.png           — распределения расстояний до ближайшей обучающей картинки
  figures/exp3_nearest_neighbors.png      — сгенерированные картинки, ближе всех к обучающим, и их соседи
  figures/exp3_trajectory.png             — x_t и оценка x̂_0 по шагам генерации

Запуск из корня репозитория (1.5–2.5 ч на RTX 3050 Laptop; нужны веса checkpoints/ema_final.pt):
    python -m scripts.exp3_generation
    python -m scripts.exp3_generation --fresh        # сгенерировать заново, не используя сохранённые картинки
При первом запуске скачиваются веса Inception-v3 (~90 МБ).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import CIFAR10

from src.diffusion import GaussianDiffusion
from src.fid import fid_from_features, from_uint8, inception_features, load_inception, nearest_neighbors, to_uint8
from src.plotting import COLORS, INK_SECONDARY, apply_style, save_figure, show_image_grid
from src.training import amp_dtype_for, load_ema_model
from src.unet import UNet
from src.utils import ensure_dir, get_device, get_logger, load_config, make_generator, resolve_path, save_json, set_seed


# --------------------------------------------------------------------------- #
# Часть А. Генерация с продолжением
# --------------------------------------------------------------------------- #
def generate_samples(diffusion, model, n, batch_size, seed, amp_dtype, sampling, checkpoint: Path,
                     cache: Path, fresh: bool, logger) -> tuple[np.ndarray, float]:
    """N картинок uint8 [N, 32, 32, 3]; у батча b свой генератор с seed + 7000 + b.

    Благодаря отдельному генератору на каждый батч результат не зависит от того, сколько раз
    генерация прерывалась. Сохранённые картинки используются, только если они получены с теми же
    весами и настройками; при увеличении N уже готовые картинки дополняются новыми.

    Returns:
        ``(картинки, минут на генерацию в этом запуске)``.
    """
    device = next(model.parameters()).device
    meta = {"checkpoint": checkpoint.name, "checkpoint_bytes": checkpoint.stat().st_size, "seed": seed,
            "batch_size": batch_size, "variance": sampling["variance"], "clip_x0": sampling["clip_x0"]}
    images = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    done = 0
    if cache.exists() and not fresh:
        with np.load(cache, allow_pickle=False) as data:
            if json.loads(str(data["meta"])) == meta:
                saved = data["images"]
                done = min(n, len(saved)) // batch_size * batch_size if len(saved) < n else n
                images[:done] = saved[:done]
                if done == n:
                    logger.info("Найдено %d сохранённых картинок — генерация не нужна", len(saved))
                else:
                    logger.info("Найдено %d сохранённых картинок, продолжаю с %d", len(saved), done)
            else:
                logger.warning("Сохранённые картинки получены с другими весами или настройками — генерирую заново")

    started = time.perf_counter()
    n_batches = math.ceil(n / batch_size)
    # done — либо N, либо число картинок в целых батчах, поэтому ceil даёт первый недоделанный батч.
    for b in range(math.ceil(done / batch_size), n_batches):
        batch_started = time.perf_counter()
        start, end = b * batch_size, min(n, (b + 1) * batch_size)
        out = diffusion.p_sample_loop(model, shape=(end - start, 3, 32, 32),
                                      generator=make_generator(seed + 7000 + b, device),
                                      variance=sampling["variance"], clip_x0=sampling["clip_x0"],
                                      autocast_dtype=amp_dtype)
        images[start:end] = to_uint8(out.images)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez(tmp, images=images[:end], meta=json.dumps(meta))
        os.replace(tmp, cache)
        sec = time.perf_counter() - batch_started
        logger.info("  батч %d/%d: %d картинок готово, %.0f с на батч, осталось ~%.0f мин",
                    b + 1, n_batches, end, sec, (n_batches - b - 1) * sec / 60)
    return images, (time.perf_counter() - started) / 60


# --------------------------------------------------------------------------- #
# Рисунки
# --------------------------------------------------------------------------- #
def plot_samples(images: np.ndarray, side: int, path: Path) -> None:
    grid = from_uint8(images[: side * side]).reshape(side, side, 3, 32, 32)
    fig = show_image_grid(grid, title=f"Картинки, сгенерированные из чистого шума ({side * side} из {len(images)})",
                          cell_size=0.75)
    save_figure(fig, path)


def plot_nn_distances(gen_d: np.ndarray, test_d: np.ndarray, path: Path) -> None:
    """Гистограммы расстояния до ближайшей обучающей картинки: сгенерированные против тестовых."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0, max(gen_d.max(), test_d.max()) * 1.02, 50)
    for d, color, label in [(test_d, COLORS["orange"], "настоящие тестовые"),
                            (gen_d, COLORS["blue"], "сгенерированные")]:
        ax.hist(d, bins=bins, histtype="step", linewidth=1.8, color=color, density=True,
                label=f"{label} (медиана {np.median(d):.3f})")
        ax.axvline(np.median(d), color=color, linewidth=1, linestyle="--")
    ax.set(title="Расстояние до ближайшей картинки из обучающей выборки",
           xlabel="RMS-расстояние по пикселям (яркости в [0, 1])", ylabel="Плотность", xlim=(0, None))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4)  # место над столбцами под легенду и подпись
    ax.legend(loc="upper right")
    ax.text(0.01, 0.97, "Копии обучающих картинок\nдали бы расстояния около 0", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=INK_SECONDARY)
    save_figure(fig, path)


def plot_nearest(gen: np.ndarray, gen_d: np.ndarray, gen_i: np.ndarray, reference: np.ndarray,
                 n_rows: int, path: Path) -> None:
    """Сгенерированные картинки, ближе всех подошедшие к обучающим, и их ближайшие обучающие соседи."""
    rows = np.argsort(gen_d[:, 0])[:n_rows]
    k = gen_i.shape[1]
    grid = torch.stack([torch.cat([from_uint8(gen[r:r + 1]), from_uint8(reference[gen_i[r]])]) for r in rows])
    fig = show_image_grid(grid, row_labels=[f"d = {gen_d[r, 0]:.3f}" for r in rows],
                          col_labels=["сгенерированная"] + [f"сосед {j + 1}" for j in range(k)],
                          title="Сгенерированные картинки, ближе всех подошедшие к обучающим", cell_size=0.95)
    save_figure(fig, path)


@torch.no_grad()
def plot_trajectory(diffusion, model, n, steps, seed, amp_dtype, sampling, path: Path) -> None:
    """Как из шума появляется картинка: x_t и оценка x̂_0 на выбранных шагах генерации."""
    device = next(model.parameters()).device
    out = diffusion.p_sample_loop(model, shape=(n, 3, 32, 32), generator=make_generator(seed + 9000, device),
                                  record_steps=steps, variance=sampling["variance"], clip_x0=sampling["clip_x0"],
                                  autocast_dtype=amp_dtype)
    rows, labels = [], []
    for i in range(n):
        rows.append(torch.stack([out.x_t[t][i] for t in steps]))
        rows.append(torch.stack([out.x0_pred[t][i] if t > 0 else out.x_t[0][i] for t in steps]))
        labels += [f"№{i + 1}: $x_t$", f"№{i + 1}: $\\hat x_0$"]
    fig = show_image_grid(torch.stack(rows), row_labels=labels, col_labels=[f"t = {t}" for t in steps],
                          title="Траектория генерации: зашумлённая картинка $x_t$ и оценка сети $\\hat x_0$",
                          cell_size=0.85)
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #
def distance_stats(d: np.ndarray) -> dict[str, float]:
    return {"median": round(float(np.median(d)), 4), "p5": round(float(np.percentile(d, 5)), 4),
            "min": round(float(d.min()), 4), "mean": round(float(d.mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="путь к конфигу")
    parser.add_argument("--checkpoint", default=None, help="веса модели вместо experiment3.checkpoint")
    parser.add_argument("--fresh", action="store_true", help="не использовать сохранённые сгенерированные картинки")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    ecfg = cfg["experiment3"]
    paths = cfg["paths"]
    logger = get_logger("exp3", log_file=resolve_path(paths["logs_dir"]) / "exp3_generation.log")
    results_dir = ensure_dir(paths["results_dir"])
    raw_dir = ensure_dir(results_dir / "raw")
    figures_dir = ensure_dir(paths["figures_dir"])
    apply_style()

    device = get_device()
    amp_dtype = amp_dtype_for(device)
    checkpoint = resolve_path(args.checkpoint or ecfg["checkpoint"])
    if not checkpoint.exists():
        raise SystemExit(f"Нет файла весов {checkpoint}. Скачайте его по ссылке из README (раздел «Веса моделей»).")
    model = load_ema_model(UNet.from_config(cfg), checkpoint, device)
    diffusion = GaussianDiffusion.from_config(cfg).to(device)
    sampling = cfg["sampling"]
    n = ecfg["n_samples"]
    logger.info("Веса: %s; устройство: %s; смешанная точность: %s", checkpoint.name, device, amp_dtype or "нет")

    # Веса Inception загружаются до генерации: если скачать их не получится, это выяснится сразу,
    # а не после двух часов генерации.
    logger.info("Загружаю сеть Inception-v3 для FID (при первом запуске скачивается ~90 МБ)")
    inception = load_inception("cpu")

    # ---- А. Генерация ----
    logger.info("Часть А: генерация %d картинок, батч %d", n, ecfg["batch_size"])
    cache = raw_dir / f"exp3_samples_{sampling['variance']}.npz"
    try:
        gen, gen_minutes = generate_samples(diffusion, model, n, ecfg["batch_size"], seed, amp_dtype, sampling,
                                            checkpoint, cache, args.fresh, logger)
    except KeyboardInterrupt:
        logger.info("Генерация остановлена. Готовые батчи сохранены в %s; повторный запуск продолжит с них.", cache)
        return
    plot_samples(gen, ecfg["grid_size"], figures_dir / "exp3_samples.png")

    # ---- Б. FID ----
    data_dir = resolve_path(paths["data_dir"])
    train = CIFAR10(root=str(data_dir), train=True, download=True).data          # uint8 [50000, 32, 32, 3]
    test = CIFAR10(root=str(data_dir), train=False, download=True).data          # uint8 [10000, 32, 32, 3]
    rng = np.random.default_rng(seed)
    reference = train if ecfg["reference_size"] >= len(train) else \
        train[np.sort(rng.choice(len(train), ecfg["reference_size"], replace=False))]
    real = test[np.sort(rng.choice(len(test), min(n, len(test)), replace=False))]

    logger.info("Часть Б: признаки Inception для %d обучающих, %d сгенерированных и %d тестовых картинок",
                len(reference), len(gen), len(real))
    started = time.perf_counter()
    inception = inception.to(device)
    bs = ecfg["inception_batch_size"]
    feats_ref = inception_features(inception, reference, bs)
    feats_gen = inception_features(inception, gen, bs)
    feats_real = inception_features(inception, real, bs)
    del inception
    fid_gen = fid_from_features(feats_gen, feats_ref)
    fid_real = fid_from_features(feats_real, feats_ref)
    logger.info("  FID сгенерированных: %.2f; FID %d настоящих тестовых (ориентир): %.2f (%.1f мин)",
                fid_gen, len(real), fid_real, (time.perf_counter() - started) / 60)

    # ---- В. Запоминание ----
    logger.info("Часть В: расстояния до ближайших обучающих картинок")
    gen_d, gen_i = nearest_neighbors(gen, reference, k=ecfg["nn_neighbors"], device=device)
    real_d, _ = nearest_neighbors(real, reference, k=1, device=device)
    gen_stats, real_stats = distance_stats(gen_d[:, 0]), distance_stats(real_d[:, 0])
    logger.info("  медиана расстояния: сгенерированные %.4f, тестовые %.4f; минимум: %.4f и %.4f",
                gen_stats["median"], real_stats["median"], gen_stats["min"], real_stats["min"])
    plot_nn_distances(gen_d[:, 0], real_d[:, 0], figures_dir / "exp3_nn_distances.png")
    plot_nearest(gen, gen_d, gen_i, reference, ecfg["nn_examples"], figures_dir / "exp3_nearest_neighbors.png")

    # ---- Г. Траектория ----
    logger.info("Часть Г: траектория генерации")
    plot_trajectory(diffusion, model, ecfg["trajectory_samples"], ecfg["trajectory_steps"], seed, amp_dtype,
                    sampling, figures_dir / "exp3_trajectory.png")

    save_json({
        "checkpoint": checkpoint.name,
        "sampling": sampling,
        "n_samples": n,
        "reference": f"CIFAR-10 train, {len(reference)} картинок",
        "fid": round(fid_gen, 3),
        "fid_real_test_same_n": round(fid_real, 3),
        "nearest_train_distance": {"generated": gen_stats, "real_test": real_stats},
        "generation_minutes_this_run": round(gen_minutes, 1),
    }, results_dir / "exp3_summary.json")
    logger.info("Готово. Рисунки: %s, таблицы: %s", figures_dir, results_dir)


if __name__ == "__main__":
    main()
