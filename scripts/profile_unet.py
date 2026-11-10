"""Проверка U-Net на конкретной видеокарте: влезает ли обучение в память и сколько оно займёт.

Делает несколько настоящих шагов обучения (прямой процесс → U-Net → MSE → Adam) на случайных
картинках, без загрузки датасета, для каждого размера батча из списка. Для каждого печатает
пиковую видеопамять, скорость и оценку времени одной эпохи (45 000 картинок). Отдельно
меряет скорость генерации: сколько займёт обратный процесс из 1000 шагов.

Используется смешанная точность: bfloat16, если видеокарта его поддерживает (RTX 30xx и новее),
иначе float16 с масштабированием градиентов.

Результаты:
    results/unet_profile.json — все замеры (зависят от видеокарты)

Запуск из корня репозитория:
    python -m scripts.profile_unet
    python -m scripts.profile_unet --batch-sizes 64 128 256 --steps 50
"""

from __future__ import annotations

import argparse
import platform
import time

import torch
import torch.nn.functional as F

from src.diffusion import GaussianDiffusion
from src.training import amp_dtype_for
from src.unet import UNet, count_parameters
from src.utils import ensure_dir, get_device, get_logger, load_config, make_generator, resolve_path, save_json, set_seed

TRAIN_SIZE = 45_000


def profile_training(cfg, diffusion, batch_size, steps, warmup, device, amp_dtype) -> dict:
    """Время и память шага обучения при заданном размере батча. При нехватке памяти — ``{"oom": True}``."""
    model = UNet.from_config(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    gen = make_generator(cfg["seed"], device)
    x0 = torch.rand(batch_size, 3, 32, 32, device=device, generator=gen) * 2 - 1

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for i in range(warmup + steps):
            if i == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
            t = diffusion.sample_timesteps(batch_size, device, generator=gen)
            noise = torch.randn(x0.shape, device=device, generator=gen)
            x_t = diffusion.q_sample(x0, t, noise=noise)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                loss = F.mse_loss(model(x_t, t).float(), noise)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        del model, optimizer
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "oom": True}

    sec_per_step = (time.perf_counter() - start) / steps
    steps_per_epoch = TRAIN_SIZE // batch_size
    result = {
        "batch_size": batch_size,
        "oom": False,
        "sec_per_step": round(sec_per_step, 4),
        "images_per_sec": round(batch_size / sec_per_step, 1),
        "steps_per_epoch": steps_per_epoch,
        "min_per_epoch": round(steps_per_epoch * sec_per_step / 60, 2),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
        "loss_is_finite": bool(torch.isfinite(loss).item()),
    }
    return result


@torch.no_grad()
def profile_sampling(cfg, diffusion, batch_size, steps, device, amp_dtype) -> dict:
    """Скорость одного шага генерации (прямой проход сети без градиентов) и оценка на 1000 шагов."""
    model = UNet.from_config(cfg).to(device).eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device, generator=make_generator(cfg["seed"], device))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for i in range(steps + 2):
            if i == 2:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                model(x, diffusion.num_steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "oom": True}
    sec_per_step = (time.perf_counter() - start) / steps
    full = sec_per_step * diffusion.num_steps
    return {
        "batch_size": batch_size,
        "oom": False,
        "sec_per_step": round(sec_per_step, 4),
        "min_per_batch_full_chain": round(full / 60, 2),
        "min_per_5000_images": round(full * 5000 / batch_size / 60, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="путь к конфигу")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128],
                        help="размеры батча для проверки обучения")
    parser.add_argument("--steps", type=int, default=30, help="шагов обучения на замер")
    parser.add_argument("--sample-batch", type=int, default=256, help="размер батча при генерации")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    paths = cfg["paths"]
    logger = get_logger("profile_unet", log_file=resolve_path(paths["logs_dir"]) / "profile_unet.log")
    device = get_device()
    amp_dtype = amp_dtype_for(device)
    diffusion = GaussianDiffusion.from_config(cfg).to(device)
    steps = args.steps if device.type == "cuda" else min(args.steps, 3)

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "нет (CPU)"
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else 0.0
    n_params = count_parameters(UNet.from_config(cfg))
    logger.info("Видеокарта: %s, память %.1f ГБ", gpu_name, gpu_mem)
    logger.info("PyTorch %s, смешанная точность: %s", torch.__version__, amp_dtype or "выключена")
    logger.info("U-Net: %.2f млн параметров", n_params / 1e6)
    if device.type != "cuda":
        logger.warning("CUDA недоступна: замеры на CPU только для проверки, что скрипт работает")

    logger.info("Обучение (%d шагов на замер):", steps)
    training = []
    for bs in args.batch_sizes:
        r = profile_training(cfg, diffusion, bs, steps, warmup=3, device=device, amp_dtype=amp_dtype)
        training.append(r)
        if r["oom"]:
            logger.info("  батч %4d: не хватает видеопамяти", bs)
        else:
            mem = f"{r['peak_memory_gb']:.2f} ГБ" if r["peak_memory_gb"] is not None else "-"
            logger.info("  батч %4d: %.3f с/шаг, %6.0f картинок/с, эпоха %.1f мин, пик памяти %s",
                        bs, r["sec_per_step"], r["images_per_sec"], r["min_per_epoch"], mem)

    sampling = profile_sampling(cfg, diffusion, args.sample_batch, steps, device, amp_dtype)
    if sampling["oom"]:
        logger.info("Генерация, батч %d: не хватает видеопамяти", args.sample_batch)
    else:
        logger.info("Генерация, батч %d: %.3f с/шаг; 1000 шагов — %.1f мин; 5000 картинок — %.0f мин",
                    args.sample_batch, sampling["sec_per_step"], sampling["min_per_batch_full_chain"],
                    sampling["min_per_5000_images"])

    out = resolve_path(paths["results_dir"]) / "unet_profile.json"
    ensure_dir(out.parent)
    save_json({
        "device": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 2),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "parameters": n_params,
        "training": training,
        "sampling": sampling,
    }, out)
    logger.info("Замеры сохранены: %s", out)


if __name__ == "__main__":
    main()
