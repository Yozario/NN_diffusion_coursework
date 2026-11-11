"""Тесты частей цикла обучения: EMA, порядок батчей, разогрев, чекпойнты."""

import pytest
import torch

from src.training import (EMA, EpochBatchSampler, amp_dtype_for, load_checkpoint, load_ema_model, save_checkpoint,
                          warmup_lr)


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
def test_ema_update_formula():
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, decay=0.9999)
    old = ema.model.weight.clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    d = ema.current_decay()
    assert d == pytest.approx(2 / 11)                      # первое обновление: (1 + 1) / (10 + 1)
    ema.update(model)
    assert torch.allclose(ema.model.weight, d * old + (1 - d) * model.weight)


def test_ema_decay_warms_up_to_target():
    ema = EMA(torch.nn.Linear(1, 1), decay=0.999)
    ema.num_updates = 10**6
    assert ema.current_decay() == 0.999


def test_ema_is_separate_copy_without_gradients():
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, decay=0.99)
    assert all(not p.requires_grad for p in ema.model.parameters())
    assert ema.model.weight.data_ptr() != model.weight.data_ptr()
    assert not ema.model.training


def test_ema_state_roundtrip():
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, decay=0.99)
    for _ in range(5):
        with torch.no_grad():
            model.weight.add_(0.1)
        ema.update(model)
    restored = EMA(torch.nn.Linear(3, 2), decay=0.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.num_updates == 5 and restored.decay == 0.99
    assert torch.equal(restored.model.weight, ema.model.weight)


# --------------------------------------------------------------------------- #
# Порядок батчей
# --------------------------------------------------------------------------- #
def test_sampler_covers_epoch_once():
    sampler = EpochBatchSampler(100, 32, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3               # неполный батч из 4 объектов отброшен
    flat = [i for b in batches for i in b]
    assert len(flat) == len(set(flat)) == 96


def test_sampler_epochs_differ_and_repeat():
    sampler = EpochBatchSampler(100, 10, seed=0)
    first = list(sampler)
    sampler.set_position(1)
    second = list(sampler)
    sampler.set_position(0)
    assert first != second
    assert list(sampler) == first


def test_sampler_resumes_mid_epoch():
    sampler = EpochBatchSampler(100, 10, seed=3)
    sampler.set_position(4)
    full = list(sampler)
    sampler.set_position(4, start_batch=6)
    assert list(sampler) == full[6:]
    assert len(sampler) == 4
    with pytest.raises(ValueError):
        sampler.set_position(4, start_batch=10)


# --------------------------------------------------------------------------- #
# Разогрев и чекпойнты
# --------------------------------------------------------------------------- #
def test_warmup_lr():
    assert warmup_lr(0, 2e-4, 1000) == pytest.approx(2e-7)
    assert warmup_lr(499, 2e-4, 1000) == pytest.approx(1e-4)
    assert warmup_lr(999, 2e-4, 1000) == pytest.approx(2e-4)
    assert warmup_lr(50_000, 2e-4, 1000) == pytest.approx(2e-4)
    assert warmup_lr(0, 2e-4, 0) == 2e-4


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, decay=0.9)
    path = tmp_path / "ckpt" / "last.pt"
    save_checkpoint({"step": 7, "model": model.state_dict(), "ema": ema.state_dict(),
                     "rng": {"torch_cpu": torch.get_rng_state(), "torch_cuda": None},
                     "config": {"seed": 42, "model": {"channel_mults": [1, 2, 2]}}}, path)
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()
    state = load_checkpoint(path)
    assert state["step"] == 7 and state["config"]["model"]["channel_mults"] == [1, 2, 2]

    restored = load_ema_model(torch.nn.Linear(3, 2), path)
    assert torch.equal(restored.weight, ema.model.weight)

    ema_only = tmp_path / "ema.pt"
    save_checkpoint({"step": 7, "model": ema.model.state_dict()}, ema_only)
    assert torch.equal(load_ema_model(torch.nn.Linear(3, 2), ema_only).weight, ema.model.weight)


def test_amp_dtype_for_cpu_and_disabled():
    assert amp_dtype_for(torch.device("cpu")) is None
    assert amp_dtype_for(torch.device("cuda"), enabled=False) is None
