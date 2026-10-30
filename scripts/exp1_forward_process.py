r"""Эксперимент 1. Прямой процесс диффузии.

Три части:
  А. Расписания шума: как убывает доля сигнала ᾱ_t и отношение сигнал/шум SNR(t)
     для линейного и косинусного расписаний.
  Б. Деградация картинок: одни и те же тестовые картинки, зашумлённые пошаговым процессом,
     на разных шагах t.
  В. Проверка эквивалентности: пошаговый процесс (t раз применить один шаг) и замкнутая
     формула x_t = sqrt(ᾱ_t)·x_0 + sqrt(1 − ᾱ_t)·ε должны давать одно и то же распределение x_t.

Как проверяется часть В. При фиксированной x_0 стандартизованный остаток
    z = (x_t − sqrt(ᾱ_t)·x_0) / sqrt(1 − ᾱ_t)
по теории распределён как N(0, 1) независимо в каждом пикселе. Для каждого расписания,
шага t и повтора (свой seed) считаются:
  * двухвыборочный тест Колмогорова — Смирнова: z пошагового процесса против z замкнутой формулы
    (H0: распределения совпадают);
  * одновыборочный тест КС: z пошагового процесса против N(0, 1);
  * дисперсия остатка x_t − sqrt(ᾱ_t)·x_0 вдоль всей траектории против теоретической 1 − ᾱ_t.
Если H0 верна, p-значения распределены равномерно на [0, 1], и примерно 5% из них меньше 0.05.

Результаты (папки из configs/default.yaml):
  figures/exp1_schedules.png            — ᾱ_t и SNR(t)
  figures/exp1_degradation_linear.png   — сетка «картинка × шаг», линейное расписание
  figures/exp1_degradation_cosine.png   — то же, косинусное
  figures/exp1_z_histograms.png         — гистограммы z для двух способов против N(0, 1)
  figures/exp1_variance_check.png       — дисперсия остатка: эксперимент против теории
  figures/exp1_pvalues.png              — эмпирическая функция распределения p-значений
  results/exp1_schedules.csv            — ᾱ_t и SNR(t) на всех шагах
  results/exp1_ks_tests.csv             — все тесты (расписание × шаг × повтор)
  results/exp1_ks_summary.csv           — сводка по расписанию и шагу
  results/exp1_summary.json             — ключевые числа для отчёта

Запуск из корня репозитория (CPU, около 2–5 минут):
    python -m scripts.exp1_forward_process
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
from scipy import stats

from src.data import CIFAR10_CLASSES_RU, fixed_examples, load_cifar10_from_config
from src.diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule
from src.plotting import COLORS, INK_SECONDARY, REFERENCE, apply_style, save_figure, show_image_grid
from src.utils import ensure_dir, get_logger, load_config, make_generator, resolve_path, save_json, set_seed

SCHEDULE_LABELS = {"linear": "линейное", "cosine": "косинусное"}
SCHEDULE_COLORS = {"linear": COLORS["blue"], "cosine": COLORS["orange"]}
ALPHA = 0.05  # уровень значимости


# --------------------------------------------------------------------------- #
# Подготовка
# --------------------------------------------------------------------------- #
def build_diffusions(cfg: dict[str, Any]) -> dict[str, GaussianDiffusion]:
    """Оба расписания с параметрами из конфига (поле ``diffusion.schedule`` здесь не важно)."""
    dcfg = cfg["diffusion"]
    num_steps = dcfg["num_steps"]
    return {
        "linear": GaussianDiffusion(linear_beta_schedule(num_steps, dcfg["beta_start"], dcfg["beta_end"])),
        "cosine": GaussianDiffusion(cosine_beta_schedule(num_steps, s=dcfg["cosine_s"])),
    }


def standardize(x_t: torch.Tensor, x0: torch.Tensor, abar: float) -> np.ndarray:
    """z = (x_t − sqrt(ᾱ_t)·x_0) / sqrt(1 − ᾱ_t), вытянутый в одномерный массив float64."""
    z = (x_t.double() - abar ** 0.5 * x0.double()) / (1.0 - abar) ** 0.5
    return z.flatten().numpy()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Список словарей с одинаковыми ключами → CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Часть А. Расписания шума
# --------------------------------------------------------------------------- #
def run_schedules(diffusions: dict[str, GaussianDiffusion], figures_dir: Path, results_dir: Path) -> dict:
    """Графики ᾱ_t и SNR(t), таблица значений и характерные шаги для отчёта."""
    num_steps = next(iter(diffusions.values())).num_steps
    t = np.arange(1, num_steps + 1)
    abar = {name: d.alphas_cumprod[1:].double().numpy() for name, d in diffusions.items()}
    snr = {name: d.snr_values[1:].double().numpy() for name, d in diffusions.items()}

    write_csv([{"t": int(k),
                **{f"alpha_bar_{n}": f"{abar[n][k - 1]:.6e}" for n in diffusions},
                **{f"snr_{n}": f"{snr[n][k - 1]:.6e}" for n in diffusions}} for k in t],
              results_dir / "exp1_schedules.csv")

    summary = {}
    for name in diffusions:
        summary[name] = {
            # первый шаг, на котором шума становится больше, чем сигнала
            "t_snr_below_1": int(t[np.argmax(snr[name] < 1.0)]),
            # первый шаг, на котором амплитуда сигнала sqrt(ᾱ_t) падает ниже 10%
            "t_signal_below_10pct": int(t[np.argmax(np.sqrt(abar[name]) < 0.1)]),
            "alpha_bar_T": float(abar[name][-1]),
            "alpha_bar_at_500": float(abar[name][499]),
        }

    fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in diffusions:
        color = SCHEDULE_COLORS[name]
        ax_a.plot(t, abar[name], color=color, label=SCHEDULE_LABELS[name])
        ax_s.plot(t, snr[name], color=color, label=SCHEDULE_LABELS[name])
        t1 = summary[name]["t_snr_below_1"]
        ax_s.plot([t1], [1.0], "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        # Подпись линейного — снизу слева от точки, косинусного — сверху справа: там нет кривых.
        below_left = name == "linear"
        ax_s.annotate(f"t = {t1}", (t1, 1.0), xytext=(-8, -14) if below_left else (8, 6),
                      textcoords="offset points", ha="right" if below_left else "left",
                      fontsize=9, color=INK_SECONDARY)

    ax_a.set(title=r"(а) Доля сигнала $\bar\alpha_t$", xlabel="Шаг t", ylabel=r"$\bar\alpha_t$",
             xlim=(0, num_steps), ylim=(0, 1.02))
    ax_s.set_yscale("log")
    ax_s.axhline(1.0, color=REFERENCE, linewidth=1, linestyle="--")
    ax_s.text(num_steps * 0.98, 1.4, "SNR = 1", ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax_s.set(title=r"(б) Отношение сигнал/шум $\bar\alpha_t\,/\,(1-\bar\alpha_t)$",
             xlabel="Шаг t", ylabel="SNR(t), логарифмическая шкала", xlim=(0, num_steps))
    ax_a.legend(title="Расписание", loc="upper right")
    save_figure(fig, figures_dir / "exp1_schedules.png")
    return summary


# --------------------------------------------------------------------------- #
# Часть Б. Деградация картинок
# --------------------------------------------------------------------------- #
def alpha_label(value: float) -> str:
    """Подпись ᾱ_t для столбца сетки: «ᾱ = 0.66» или «ᾱ < 0.01»."""
    return r"$\bar\alpha$ = " + f"{value:.2f}" if value >= 0.01 else r"$\bar\alpha$ < 0.01"


def run_degradation(
    diffusions: dict[str, GaussianDiffusion],
    images: torch.Tensor,
    labels: torch.Tensor,
    timesteps: list[int],
    seed: int,
    figures_dir: Path,
) -> None:
    """Одна траектория пошагового процесса для каждой картинки, снимки на шагах ``timesteps``.

    Для обоих расписаний генератор шума стартует с одного seed, поэтому на первых шагах
    картинки получают одинаковый шум, и различие рисунков — только эффект расписания.
    """
    row_labels = [CIFAR10_CLASSES_RU[int(c)] for c in labels]
    for name, diffusion in diffusions.items():
        _, saved = diffusion.q_sample_iterative(images, max(timesteps), generator=make_generator(seed),
                                                record=timesteps)
        grid = torch.stack([saved[t] for t in timesteps], dim=1)  # [N, len(timesteps), 3, H, W]
        col_labels = [f"t = {t}\n" + alpha_label(diffusion.alphas_cumprod[t].item()) for t in timesteps]
        fig = show_image_grid(grid, row_labels=row_labels, col_labels=col_labels,
                              title=f"Прямой процесс, {SCHEDULE_LABELS[name]} расписание шума")
        save_figure(fig, figures_dir / f"exp1_degradation_{name}.png")


# --------------------------------------------------------------------------- #
# Часть В. Пошаговый процесс против замкнутой формулы
# --------------------------------------------------------------------------- #
def run_equivalence(
    diffusions: dict[str, GaussianDiffusion],
    x0: torch.Tensor,
    ks_timesteps: list[int],
    repeats: int,
    variance_every: int,
    seed: int,
    logger,
) -> tuple[list[dict], dict, dict]:
    """KS-тесты на шагах ``ks_timesteps`` и дисперсия остатка вдоль всей траектории.

    Returns:
        ``(tests, variance, z_examples)``:
        tests — по строке на расписание × шаг × повтор;
        variance — для каждого расписания шаги, эмпирическая и теоретическая дисперсии
        (среднее по повторам);
        z_examples — массивы z обоих способов на шагах ``ks_timesteps`` (повтор 0,
        линейное расписание) для гистограмм.
    """
    tests: list[dict] = []
    variance: dict[str, dict[str, np.ndarray]] = {}
    z_examples: dict[int, dict[str, np.ndarray]] = {}
    x0_64 = x0.double()

    for s_idx, (name, diffusion) in enumerate(diffusions.items()):
        num_steps = diffusion.num_steps
        var_steps = list(range(variance_every, num_steps + 1, variance_every))
        record = sorted(set(ks_timesteps) | set(var_steps))
        abar = diffusion.alphas_cumprod.double()
        var_sum = np.zeros(len(var_steps))
        started = time.perf_counter()

        for r in range(repeats):
            # Разные seed для двух способов и для каждого повтора: выборки независимы.
            gen_iter = make_generator(seed + 10_000 * (s_idx + 1) + r)
            gen_closed = make_generator(seed + 10_000 * (s_idx + 1) + 5_000 + r)
            _, saved = diffusion.q_sample_iterative(x0, num_steps, generator=gen_iter, record=record)

            for k, t in enumerate(var_steps):
                residual = saved[t].double() - abar[t].sqrt() * x0_64
                var_sum[k] += residual.var().item()

            for t in ks_timesteps:
                a = abar[t].item()
                z_iter = standardize(saved[t], x0, a)
                z_closed = standardize(diffusion.q_sample(x0, t, generator=gen_closed), x0, a)
                two_sample = stats.ks_2samp(z_iter, z_closed)
                vs_normal = stats.kstest(z_iter, "norm")
                tests.append({
                    "schedule": name, "t": t, "repeat": r, "alpha_bar": a, "n_values": z_iter.size,
                    "ks2_stat": two_sample.statistic, "ks2_pvalue": two_sample.pvalue,
                    "ks_norm_stat": vs_normal.statistic, "ks_norm_pvalue": vs_normal.pvalue,
                    "z_mean_iter": z_iter.mean(), "z_std_iter": z_iter.std(),
                    "z_mean_closed": z_closed.mean(), "z_std_closed": z_closed.std(),
                })
                if name == "linear" and r == 0:
                    z_examples[t] = {"iter": z_iter, "closed": z_closed}
            del saved

        variance[name] = {
            "t": np.array(var_steps),
            "empirical": var_sum / repeats,
            "theory": (1.0 - abar[var_steps]).numpy(),
        }
        logger.info("  %s: %d повторов за %.1f с", SCHEDULE_LABELS[name], repeats, time.perf_counter() - started)
    return tests, variance, z_examples


def summarize_tests(tests: list[dict]) -> list[dict]:
    """Сводка по каждому расписанию и шагу: средние статистики и доля отвергнутых H0."""
    rows = []
    for name in SCHEDULE_LABELS:
        for t in sorted({row["t"] for row in tests}):
            group = [row for row in tests if row["schedule"] == name and row["t"] == t]
            if not group:
                continue
            col = lambda key: np.array([row[key] for row in group])  # noqa: E731
            rows.append({
                "schedule": name, "t": t, "repeats": len(group),
                "ks2_stat_mean": round(col("ks2_stat").mean(), 5),
                "ks2_reject_share": round((col("ks2_pvalue") < ALPHA).mean(), 3),
                "ks_norm_reject_share": round((col("ks_norm_pvalue") < ALPHA).mean(), 3),
                "z_mean_iter": round(col("z_mean_iter").mean(), 5),
                "z_std_iter": round(col("z_std_iter").mean(), 5),
                "z_std_closed": round(col("z_std_closed").mean(), 5),
            })
    return rows


def plot_z_histograms(z_examples: dict[int, dict[str, np.ndarray]], figures_dir: Path) -> None:
    """Гистограммы z для двух способов на нескольких шагах против плотности N(0, 1)."""
    steps = sorted(z_examples)
    fig, axes = plt.subplots(1, len(steps), figsize=(2.3 * len(steps), 2.8), sharey=True)
    grid = np.linspace(-4.5, 4.5, 301)
    bins = np.linspace(-4.5, 4.5, 61)
    for ax, t in zip(np.atleast_1d(axes), steps):
        ax.hist(z_examples[t]["iter"], bins=bins, density=True, histtype="step", linewidth=1.6,
                color=COLORS["blue"], label="пошаговый")
        ax.hist(z_examples[t]["closed"], bins=bins, density=True, histtype="step", linewidth=1.6,
                color=COLORS["orange"], linestyle=(0, (4, 2)), label="замкнутая формула")
        ax.plot(grid, stats.norm.pdf(grid), color=REFERENCE, linewidth=1, label="N(0, 1)")
        ax.set(title=f"t = {t}", xlabel="z", xlim=(-4.5, 4.5))
    np.atleast_1d(axes)[0].set_ylabel("Плотность")
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Стандартизованный остаток z, линейное расписание", y=1.16, fontweight="bold")
    save_figure(fig, figures_dir / "exp1_z_histograms.png")


def plot_variance_check(variance: dict, n_values: int, repeats: int, figures_dir: Path) -> dict:
    """(а) Дисперсия остатка против теории 1 − ᾱ_t; (б) относительное отклонение в процентах.

    Серая полоса на панели (б) — ±2 стандартные ошибки выборочной дисперсии:
    для нормальной выборки из n значений относительная ошибка равна sqrt(2 / (n − 1)),
    при усреднении по повторам она уменьшается ещё в sqrt(число повторов) раз.
    """
    rel_se = np.sqrt(2.0 / (n_values - 1)) / np.sqrt(repeats)
    fig, (ax_v, ax_e) = plt.subplots(1, 2, figsize=(10, 3.6))
    deviation = {}
    for name, v in variance.items():
        color = SCHEDULE_COLORS[name]
        ax_v.plot(v["t"], v["theory"], color=color, linewidth=1.2, label=f"{SCHEDULE_LABELS[name]}: теория")
        ax_v.plot(v["t"][4::5], v["empirical"][4::5], "o", color=color, markerfacecolor="white",
                  markeredgewidth=1.4, label=f"{SCHEDULE_LABELS[name]}: эксперимент")
        rel = (v["empirical"] / v["theory"] - 1.0) * 100.0
        deviation[name] = float(np.abs(rel).max())
        ax_e.plot(v["t"], rel, color=color, linewidth=1.4, label=SCHEDULE_LABELS[name])

    ax_e.axhspan(-200 * rel_se, 200 * rel_se, color="#e2e0da", zorder=0, label="±2 станд. ошибки")
    ax_e.axhline(0, color=REFERENCE, linewidth=1)
    ax_v.set(title=r"(а) Дисперсия остатка $x_t-\sqrt{\bar\alpha_t}\,x_0$", xlabel="Шаг t",
             ylabel="Дисперсия", xlim=(0, None), ylim=(0, 1.05))
    ax_e.set(title="(б) Отклонение эксперимента от теории", xlabel="Шаг t",
             ylabel="Отклонение, %", xlim=(0, None))
    ax_v.legend(loc="lower right")
    # Легенда под осями: кривые отклонения занимают всю высоту панели.
    ax_e.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    save_figure(fig, figures_dir / "exp1_variance_check.png")
    return {"max_abs_relative_deviation_pct": deviation, "two_se_band_pct": float(200 * rel_se)}


def plot_pvalues(tests: list[dict], figures_dir: Path) -> dict:
    """Эмпирическая функция распределения p-значений двухвыборочного теста против Uniform(0, 1).

    Если пошаговый процесс и замкнутая формула дают одно распределение, кривые лежат
    около диагонали. Совпадение проверяется ещё одним тестом КС — против равномерного закона.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle="--", label="равномерное распределение")
    meta = {}
    for name in SCHEDULE_LABELS:
        p = np.sort([row["ks2_pvalue"] for row in tests if row["schedule"] == name])
        ecdf = np.arange(1, len(p) + 1) / len(p)
        uniformity = float(stats.kstest(p, "uniform").pvalue)
        ax.step(p, ecdf, where="post", color=SCHEDULE_COLORS[name],
                label=f"{SCHEDULE_LABELS[name]} ({len(p)} тестов)")
        meta[name] = {"n_tests": int(len(p)), "uniformity_ks_pvalue": uniformity,
                      "reject_share": float((p < ALPHA).mean())}
    # Итог проверки равномерности — в свободном левом верхнем углу (кривые идут вдоль диагонали).
    ax.text(0.03, 0.97, "Тест КС на равномерность p-значений:\n"
            + "\n".join(f"{SCHEDULE_LABELS[n]}: p = {m['uniformity_ks_pvalue']:.2f}" for n, m in meta.items()),
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK_SECONDARY)
    ax.set(title="p-значения двухвыборочного теста КС", xlabel="p-значение",
           ylabel="Доля тестов с меньшим p", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    save_figure(fig, figures_dir / "exp1_pvalues.png")
    return meta


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="путь к конфигу")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    ecfg = cfg["experiment1"]
    paths = cfg["paths"]
    logger = get_logger("exp1", log_file=resolve_path(paths["logs_dir"]) / "exp1_forward_process.log")
    results_dir = ensure_dir(paths["results_dir"])
    figures_dir = ensure_dir(paths["figures_dir"])
    apply_style()

    diffusions = build_diffusions(cfg)
    splits = load_cifar10_from_config(cfg)

    logger.info("Часть А: расписания шума")
    schedule_summary = run_schedules(diffusions, figures_dir, results_dir)
    for name, s in schedule_summary.items():
        logger.info("  %s: SNR < 1 с шага %d, сигнал < 10%% с шага %d, alpha_bar(500) = %.3f, alpha_bar(T) = %.2e",
                    SCHEDULE_LABELS[name], s["t_snr_below_1"], s["t_signal_below_10pct"],
                    s["alpha_bar_at_500"], s["alpha_bar_T"])

    logger.info("Часть Б: сетка деградации")
    images, labels = fixed_examples(splits.test, n=ecfg["n_images"], seed=seed, one_per_class=True)
    run_degradation(diffusions, images, labels, ecfg["grid_timesteps"], seed, figures_dir)

    logger.info("Часть В: пошаговый процесс против замкнутой формулы")
    x0, _ = fixed_examples(splits.test, n=ecfg["ks_images"], seed=seed + 1, one_per_class=False)
    tests, variance, z_examples = run_equivalence(
        diffusions, x0, ecfg["ks_timesteps"], ecfg["ks_repeats"], ecfg["variance_every"], seed, logger)
    write_csv(tests, results_dir / "exp1_ks_tests.csv")
    summary_rows = summarize_tests(tests)
    write_csv(summary_rows, results_dir / "exp1_ks_summary.csv")
    for row in summary_rows:
        logger.info("  %-10s t=%4d | D=%.4f | H0 отвергнута в %.0f%% | std(z)=%.4f",
                    SCHEDULE_LABELS[row["schedule"]], row["t"], row["ks2_stat_mean"],
                    100 * row["ks2_reject_share"], row["z_std_iter"])

    plot_z_histograms(z_examples, figures_dir)
    variance_summary = plot_variance_check(variance, x0.numel(), ecfg["ks_repeats"], figures_dir)
    pvalue_summary = plot_pvalues(tests, figures_dir)
    for name, m in pvalue_summary.items():
        logger.info("  %s: %d тестов, H0 отвергнута в %.1f%%, p равномерности p-значений = %.3f",
                    SCHEDULE_LABELS[name], m["n_tests"], 100 * m["reject_share"], m["uniformity_ks_pvalue"])
    logger.info("  Макс. отклонение дисперсии от теории: %s (полоса 2SE = +/-%.2f%%)",
                {k: f"{v:.2f}%" for k, v in variance_summary["max_abs_relative_deviation_pct"].items()},
                variance_summary["two_se_band_pct"])

    save_json({
        "seed": seed,
        "num_steps": cfg["diffusion"]["num_steps"],
        "schedules": schedule_summary,
        "ks": {"alpha": ALPHA, "values_per_test": int(x0.numel()), "repeats": ecfg["ks_repeats"],
               "timesteps": ecfg["ks_timesteps"], "per_schedule": pvalue_summary},
        "variance_check": variance_summary,
    }, results_dir / "exp1_summary.json")
    logger.info("Готово. Рисунки: %s, таблицы: %s", figures_dir, results_dir)


if __name__ == "__main__":
    main()
