"""Общие утилиты проекта: воспроизводимость, конфиги, пути, логирование.

Все скрипты начинают работу одинаково::

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("exp1", log_file=resolve_path(cfg["paths"]["logs_dir"]) / "exp1.log")
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Корень репозитория: src/utils.py -> src -> корень
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# Воспроизводимость
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True, strict: bool = False) -> None:
    """Фиксирует все источники случайности: random, numpy, torch (CPU и CUDA).

    Вызывать в самом начале скрипта, до создания моделей и обращения к GPU.

    Args:
        seed: начальное состояние генераторов.
        deterministic: выключить недетерминированные алгоритмы cuDNN
            (свёртки становятся чуть медленнее, зато результаты повторяются).
        strict: дополнительно включить ``torch.use_deterministic_algorithms``.
            Тогда PyTorch выдаёт ошибку на любой недетерминированной операции.
            По умолчанию выключено: часть операций на GPU (например, обратный
            проход некоторых видов интерполяции) не имеет детерминированной версии.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Нужна для детерминированных матричных умножений cuBLAS;
        # действует, только если задана до первой операции на GPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict:
        torch.use_deterministic_algorithms(True)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Создаёт отдельный генератор случайных чисел torch с заданным seed.

    Отдельный генератор не зависит от глобального состояния: результат функции,
    которой он передан, не меняется, если где-то раньше в коде добавили
    ещё один вызов ``torch.randn``.
    """
    return torch.Generator(device=device).manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` для DataLoader: фиксирует numpy и random в каждом процессе загрузки.

    PyTorch сам задаёт каждому процессу seed torch на основе генератора DataLoader,
    а numpy и random нужно выставить вручную (рецепт из документации PyTorch).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Конфиги и пути
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Читает YAML-конфиг. Без аргумента — ``configs/default.yaml``."""
    path = resolve_path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Сохраняет конфиг рядом с результатами, чтобы знать, с какими параметрами они получены."""
    path = resolve_path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def resolve_path(path: str | Path) -> Path:
    """Относительный путь считается от корня репозитория, абсолютный — как есть.

    Благодаря этому скрипты работают одинаково, из какой бы папки их ни запустили.
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """Создаёт папку (вместе с родительскими), если её нет, и возвращает путь к ней."""
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    """Сохраняет словарь с результатами в JSON с отступами и кириллицей без экранирования."""
    path = resolve_path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Логирование и устройство
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Логгер, который пишет в консоль и, если указан файл, ещё и в файл.

    Повторный вызов с тем же именем не дублирует обработчики.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file = resolve_path(log_file)
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def get_device(prefer_cuda: bool = True) -> torch.device:
    """GPU, если он доступен и разрешён, иначе CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
