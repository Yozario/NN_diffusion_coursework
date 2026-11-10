"""Обучение диффузионной модели (U-Net, предсказывающий шум) на CIFAR-10.

Один шаг обучения: батч картинок x_0 → случайные шаги t и шум ε → x_t по замкнутой формуле →
U-Net предсказывает ε → среднеквадратичная ошибка → Adam. После каждого шага обновляется
EMA-копия весов; именно она используется для генерации.

Что сохраняется (пути из configs/default.yaml):
    checkpoints/last.pt                    — полное состояние для продолжения (каждые checkpoint_every шагов)
    checkpoints/ema_step*.pt               — EMA-веса на промежуточных шагах
    checkpoints/ema_final.pt               — EMA-веса в конце обучения
    results/train_log.csv                  — потеря, норма градиента, скорость (каждые log_every шагов)
    results/train_val.csv                  — потеря на валидации для текущих и EMA-весов
    results/figures/train_loss.png         — кривые обучения
    results/figures/train_samples/step_*.png — картинки, сгенерированные EMA-весами из одного и того же шума

Запуск из корня репозитория:
    python -m scripts.train_diffusion                              # обучение с нуля
    python -m scripts.train_diffusion --total-steps 2000           # короткий отладочный прогон
    python -m scripts.train_diffusion --resume                     # продолжить с checkpoints/last.pt
    python -m scripts.train_diffusion --fresh                      # начать заново, удалив прошлые результаты
    python -m scripts.train_diffusion --plot-only                  # перерисовать график по сохранённым журналам

Ctrl+C останавливает обучение и сохраняет чекпойнт; продолжить — с флагом --resume.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_cifar10_from_config
from src.diffusion import GaussianDiffusion
from src.plotting import COLORS, INK_SECONDARY, apply_style, save_figure, show_image_grid
from src.training import EMA, EpochBatchSampler, amp_dtype_for, load_checkpoint, save_checkpoint, warmup_lr
from src.unet import UNet, count_parameters
from src.utils import ensure_dir, get_device, get_logger, load_config, make_generator, resolve_path, set_seed

TRAIN_LOG_FIELDS = ["step", "epoch", "lr", "loss", "grad_norm", "sec_per_step", "images_per_sec", "elapsed_min"]
VAL_LOG_FIELDS = ["step", "val_loss", "val_loss_ema"]


# --------------------------------------------------------------------------- #
# Журналы и рисунки
# --------------------------------------------------------------------------- #
class CsvLog:
    """CSV-файл, в который строки дописываются по одной (файл сразу сбрасывается на диск)."""

    def __init__(self, path: Path, fields: list[str], append: bool) -> None:
        self.path, self.fields = path, fields
        if not append or not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def write(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)

    def read(self) -> list[dict[str, float]]:
        with open(self.path, encoding="utf-8") as f:
            return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def drop_rows_after(log: CsvLog, step: int) -> None:
    """При продолжении убирает строки журнала, записанные после сохранённого чекпойнта."""
    with open(log.path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["step"]) <= step]
    with open(log.path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log.fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(train_log: CsvLog, val_log: CsvLog, path: Path, zoom_from: int = 2000,
                smooth_steps: int = 2000) -> None:
    """Кривые обучения на двух панелях.

    (а) Весь диапазон в логарифмическом масштабе: за первую тысячу шагов потеря падает
        с ~1 до ~0.035, и на одной оси с этим падением дальнейшее улучшение не видно.
    (б) С шага ``zoom_from`` в обычном масштабе: медленное снижение потери на валидации
        и разница между текущими и EMA-весами. Потеря на обучении здесь сглажена
        скользящим средним за ``smooth_steps`` шагов, иначе её колебания закрывают тренд.
    """
    train, val = train_log.read(), val_log.read()
    if not train:
        return
    steps = np.array([r["step"] for r in train])
    loss = np.array([r["loss"] for r in train])
    interval = int(steps[1] - steps[0]) if len(steps) > 1 else int(steps[0])
    val_steps = np.array([r["step"] for r in val])
    val_loss = np.array([r["val_loss"] for r in val])
    val_ema = np.array([r["val_loss_ema"] for r in val])
    show_zoom = val_steps.size > 1 and val_steps.max() > zoom_from

    fig, axes = plt.subplots(1, 2 if show_zoom else 1, figsize=(11.5 if show_zoom else 6.5, 4.2), squeeze=False)
    ax_full = axes[0][0]
    ax_full.plot(steps, loss, color=COLORS["blue"], linewidth=1, label=f"обучение (среднее за {interval} шагов)")
    if val:
        ax_full.plot(val_steps, val_loss, color=COLORS["orange"], linewidth=1.6, label="валидация, текущие веса")
        ax_full.plot(val_steps, val_ema, color=COLORS["aqua"], linewidth=1.6, label="валидация, EMA-веса")
    ax_full.set_yscale("log")
    ax_full.set(title="(а) Весь диапазон", xlabel="Шаг обучения", ylabel="MSE, логарифмическая шкала",
                xlim=(0, None))
    ax_full.legend(loc="upper right")

    if show_zoom:
        ax = axes[0][1]
        # Сглаживаем только точки после zoom_from: иначе в первое окно попадёт начальное падение потери.
        window = max(1, smooth_steps // interval)
        tail = steps > zoom_from
        if tail.sum() >= window:
            smooth = np.convolve(loss[tail], np.ones(window) / window, mode="valid")
            ax.plot(steps[tail][window - 1:], smooth, color=COLORS["blue"], linewidth=1.2,
                    label=f"обучение (скользящее среднее за {window * interval} шагов)")
        keep = val_steps >= zoom_from
        ax.plot(val_steps[keep], val_loss[keep], "o-", color=COLORS["orange"], markersize=3, linewidth=1.4,
                label="валидация, текущие веса")
        ax.plot(val_steps[keep], val_ema[keep], "s-", color=COLORS["aqua"], markersize=3, linewidth=1.4,
                label="валидация, EMA-веса")
        # Подписи итоговых значений справа от последних точек.
        for y, color, va in [(val_loss[-1], COLORS["orange"], "bottom"), (val_ema[-1], COLORS["aqua"], "top")]:
            ax.annotate(f"{y:.4f}", (val_steps[-1], y), xytext=(6, 0), textcoords="offset points",
                        va=va, fontsize=9, color=INK_SECONDARY)
        zoom_text = f"{zoom_from:,}".replace(",", " ")  # 2000 -> «2 000»
        ax.set(title=f"(б) С шага {zoom_text}, обычная шкала", xlabel="Шаг обучения",
               ylabel="MSE", xlim=(zoom_from, val_steps.max() * 1.06))
        ax.legend(loc="upper right")

    fig.suptitle("Функция потерь: MSE между настоящим и предсказанным шумом", fontweight="bold")
    save_figure(fig, path)


def save_sample_grid(images: torch.Tensor, step: int, path: Path) -> None:
    """Сетка сгенерированных картинок (число картинок — квадрат целого)."""
    n = images.shape[0]
    side = int(n ** 0.5)
    grid = images[: side * side].reshape(side, side, *images.shape[1:])
    step_text = f"{step:,}".replace(",", " ")  # 100000 -> «100 000»
    fig = show_image_grid(grid, title=f"Генерация EMA-весами, шаг обучения {step_text}", cell_size=0.8)
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Валидация
# --------------------------------------------------------------------------- #
def fixed_validation_set(dataset, n: int, seed: int, device: torch.device, num_steps: int):
    """Первые ``n`` картинок val и фиксированные для них шаги t и шум ε.

    Так потеря на валидации зависит только от весов модели, и её значения на разных
    этапах обучения можно сравнивать между собой.
    """
    n = min(n, len(dataset))
    x0 = torch.stack([dataset[i][0] for i in range(n)])
    gen = make_generator(seed + 1)
    t = torch.randint(1, num_steps + 1, (n,), generator=gen)
    noise = torch.randn(x0.shape, generator=gen)
    return x0.to(device), t.to(device), noise.to(device)


@torch.no_grad()
def validation_loss(diffusion, model, val, batch_size: int, amp_dtype) -> float:
    x0, t, noise = val
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(0, x0.shape[0], batch_size):
        with torch.autocast(x0.device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss = diffusion.training_loss(model, x0[i:i + batch_size], t=t[i:i + batch_size],
                                           noise=noise[i:i + batch_size])
        total += loss.item() * x0[i:i + batch_size].shape[0]
    model.train(was_training)
    return total / x0.shape[0]


# --------------------------------------------------------------------------- #
# Обучение
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="путь к конфигу")
    parser.add_argument("--total-steps", type=int, default=None, help="число шагов вместо training.total_steps")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="продолжить с checkpoints/last.pt")
    mode.add_argument("--fresh", action="store_true", help="начать заново, удалив чекпойнты и журналы прошлого запуска")
    mode.add_argument("--plot-only", action="store_true",
                      help="только перерисовать график потерь по сохранённым журналам, без обучения")
    return parser.parse_args()


def clear_previous_run(ckpt_dir: Path, results_dir: Path, figures_dir: Path, logger) -> None:
    """Удаляет файлы прошлого запуска этого скрипта (и только их)."""
    removed = []
    for pattern_dir, pattern in [(ckpt_dir, "last.pt"), (ckpt_dir, "ema_*.pt"),
                                 (results_dir, "train_log.csv"), (results_dir, "train_val.csv"),
                                 (figures_dir, "train_loss.png"), (figures_dir / "train_samples", "step_*.png")]:
        for p in pattern_dir.glob(pattern):
            p.unlink()
            removed.append(p.name)
    if removed:
        logger.info("Удалены результаты прошлого запуска: %d файлов", len(removed))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    if args.total_steps is not None:
        tcfg["total_steps"] = args.total_steps
    total_steps = tcfg["total_steps"]
    seed = cfg["seed"]
    set_seed(seed)

    paths = cfg["paths"]
    ckpt_dir = ensure_dir(paths["checkpoints_dir"])
    results_dir = ensure_dir(paths["results_dir"])
    figures_dir = ensure_dir(paths["figures_dir"])
    samples_dir = ensure_dir(figures_dir / "train_samples")
    last_path = ckpt_dir / "last.pt"
    if args.plot_only:
        apply_style()
        train_log = CsvLog(results_dir / "train_log.csv", TRAIN_LOG_FIELDS, append=True)
        val_log = CsvLog(results_dir / "train_val.csv", VAL_LOG_FIELDS, append=True)
        plot_curves(train_log, val_log, figures_dir / "train_loss.png")
        print(f"График сохранён: {figures_dir / 'train_loss.png'}")
        return
    logger = get_logger("train", log_file=resolve_path(paths["logs_dir"]) / "train_diffusion.log")

    if last_path.exists() and not (args.resume or args.fresh):
        logger.error("Найден чекпойнт прошлого обучения: %s. Запустите с --resume, чтобы продолжить, "
                     "или с --fresh, чтобы начать заново (прошлые чекпойнты и журналы будут удалены).", last_path)
        sys.exit(1)
    if args.resume and not last_path.exists():
        logger.error("Нет чекпойнта %s — продолжать нечего", last_path)
        sys.exit(1)
    if args.fresh:
        clear_previous_run(ckpt_dir, results_dir, figures_dir, logger)
    apply_style()

    # ---- Устройство, данные, модель ----
    device = get_device()
    amp_dtype = amp_dtype_for(device, tcfg["amp"])
    batch_size = cfg["data"]["batch_size"]

    splits = load_cifar10_from_config(cfg)
    sampler = EpochBatchSampler(len(splits.train), batch_size, seed)
    # Свой генератор у загрузчика: иначе каждый новый проход по нему берёт число из глобального
    # генератора torch, и после продолжения обучения случайные отражения картинок шли бы иначе.
    loader = DataLoader(splits.train, batch_sampler=sampler, num_workers=tcfg["num_workers"],
                        pin_memory=device.type == "cuda", persistent_workers=tcfg["num_workers"] > 0,
                        generator=make_generator(seed))
    diffusion = GaussianDiffusion.from_config(cfg).to(device)
    val = fixed_validation_set(splits.val, tcfg["val_images"], seed, device, diffusion.num_steps)

    model = UNet.from_config(cfg).to(device)
    ema = EMA(model, tcfg["ema_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    gen = make_generator(seed, device)  # шаги t и шум ε для обучения

    # Шум x_T для контрольных картинок — один и тот же на всём обучении:
    # видно, как из одного шума с ростом числа шагов получаются всё более правдоподобные картинки.
    sample_noise = torch.randn(tcfg["sample_count"], 3, 32, 32, device=device,
                               generator=make_generator(seed + 2, device))

    step, epoch, batch_in_epoch, elapsed = 0, 0, 0, 0.0
    if args.resume:
        state = load_checkpoint(last_path, device)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        gen.set_state(state["rng"]["train_generator"].cpu())
        torch.set_rng_state(state["rng"]["torch_cpu"].cpu())
        if device.type == "cuda" and state["rng"].get("torch_cuda") is not None:
            torch.cuda.set_rng_state(state["rng"]["torch_cuda"].cpu())
        step, epoch, batch_in_epoch, elapsed = state["step"], state["epoch"], state["batch_in_epoch"], state["elapsed"]
        logger.info("Продолжаю с шага %d (эпоха %d, батч %d)", step, epoch, batch_in_epoch)

    train_log = CsvLog(results_dir / "train_log.csv", TRAIN_LOG_FIELDS, append=args.resume)
    val_log = CsvLog(results_dir / "train_val.csv", VAL_LOG_FIELDS, append=args.resume)
    if args.resume:
        drop_rows_after(train_log, step)
        drop_rows_after(val_log, step)

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    logger.info("Устройство: %s, смешанная точность: %s", gpu, amp_dtype or "выключена")
    logger.info("U-Net: %.2f млн параметров; батч %d; %d батчей в эпохе; всего шагов %d",
                count_parameters(model) / 1e6, batch_size, sampler.batches_per_epoch, total_steps)

    def checkpoint_state() -> dict[str, Any]:
        # Позиция следующего батча; если эпоха только что закончилась — начало следующей.
        ep, b = (epoch + 1, 0) if batch_in_epoch >= sampler.batches_per_epoch else (epoch, batch_in_epoch)
        return {
            "step": step, "epoch": ep, "batch_in_epoch": b, "elapsed": elapsed,
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "rng": {"train_generator": gen.get_state(), "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
            "config": cfg,
        }

    def save_ema(name: str) -> None:
        save_checkpoint({"step": step, "model": ema.model.state_dict(), "config": cfg}, ckpt_dir / name)

    def generate_samples() -> None:
        started = time.perf_counter()
        out = diffusion.p_sample_loop(ema.model, x_start=sample_noise, generator=make_generator(seed + 3, device),
                                      variance=cfg["sampling"]["variance"], clip_x0=cfg["sampling"]["clip_x0"],
                                      autocast_dtype=amp_dtype)
        save_sample_grid(out.images.cpu(), step, samples_dir / f"step_{step:06d}.png")
        logger.info("Сгенерировано %d картинок за %.0f с", sample_noise.shape[0], time.perf_counter() - started)

    def validate() -> None:
        loss = validation_loss(diffusion, model, val, batch_size, amp_dtype)
        loss_ema = validation_loss(diffusion, ema.model, val, batch_size, amp_dtype)
        val_log.write({"step": step, "val_loss": f"{loss:.6f}", "val_loss_ema": f"{loss_ema:.6f}"})
        plot_curves(train_log, val_log, figures_dir / "train_loss.png")
        logger.info("Валидация на шаге %d: MSE %.5f, с EMA-весами %.5f", step, loss, loss_ema)

    # ---- Цикл обучения ----
    model.train()
    loss_sum = torch.zeros((), device=device)
    grad_sum = torch.zeros((), device=device)
    interval_start, interval_steps = time.perf_counter(), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        while step < total_steps:
            sampler.set_position(epoch, batch_in_epoch)
            for x0, _ in loader:
                x0 = x0.to(device, non_blocking=True)
                lr = warmup_lr(step, tcfg["lr"], tcfg["warmup_steps"])
                for group in optimizer.param_groups:
                    group["lr"] = lr

                with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    loss = diffusion.training_loss(model, x0, generator=gen)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                step += 1
                batch_in_epoch += 1
                interval_steps += 1
                loss_sum += loss.detach()
                grad_sum += grad_norm.detach().float()

                if step % tcfg["log_every"] == 0 or step == total_steps:
                    now = time.perf_counter()
                    sec_per_step = (now - interval_start) / interval_steps
                    elapsed += now - interval_start
                    mean_loss = (loss_sum / interval_steps).item()
                    train_log.write({
                        "step": step, "epoch": epoch, "lr": f"{lr:.3e}", "loss": f"{mean_loss:.6f}",
                        "grad_norm": f"{(grad_sum / interval_steps).item():.4f}",
                        "sec_per_step": f"{sec_per_step:.4f}",
                        "images_per_sec": f"{batch_size / sec_per_step:.1f}",
                        "elapsed_min": f"{elapsed / 60:.2f}",
                    })
                    eta_min = (total_steps - step) * sec_per_step / 60
                    logger.info("шаг %6d/%d | эпоха %3d | loss %.5f | lr %.1e | %.3f с/шаг | осталось ~%.0f мин",
                                step, total_steps, epoch, mean_loss, lr, sec_per_step, eta_min)
                    loss_sum.zero_()
                    grad_sum.zero_()
                    interval_start, interval_steps = time.perf_counter(), 0

                if step % tcfg["val_every"] == 0:
                    validate()
                if step % tcfg["sample_every"] == 0 and step != total_steps:
                    generate_samples()
                if step % tcfg["ema_save_every"] == 0:
                    save_ema(f"ema_step{step:06d}.pt")
                if step % tcfg["checkpoint_every"] == 0:
                    save_checkpoint(checkpoint_state(), last_path)
                if step >= total_steps:
                    break
                if step % tcfg["log_every"] == 0:
                    interval_start = time.perf_counter()  # валидация и генерация не входят в скорость обучения
            else:
                epoch, batch_in_epoch = epoch + 1, 0
                continue
            break
    except KeyboardInterrupt:
        save_checkpoint(checkpoint_state(), last_path)
        logger.info("Обучение остановлено на шаге %d, чекпойнт сохранён. Продолжить: "
                    "python -m scripts.train_diffusion --resume", step)
        return

    # ---- Конец обучения ----
    save_checkpoint(checkpoint_state(), last_path)
    save_ema("ema_final.pt")
    if step % tcfg["val_every"] != 0:
        validate()
    generate_samples()
    if device.type == "cuda":
        logger.info("Пик видеопамяти: %.2f ГБ", torch.cuda.max_memory_allocated(device) / 2**30)
    logger.info("Обучение завершено: %d шагов за %.1f мин. EMA-веса: %s",
                step, elapsed / 60, ckpt_dir / "ema_final.pt")


if __name__ == "__main__":
    main()
