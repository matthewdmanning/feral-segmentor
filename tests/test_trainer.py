"""Standalone tests for the dependency-injected Trainer.

These intentionally use dummies/DI so they pass even while the real
collaborators (build_model, segmentation_loss, SegmentationDataset) are
skeletons. We never import those here.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from feral_segmentor.training.trainer import Trainer


def _make_cfg(epochs: int) -> SimpleNamespace:
    return SimpleNamespace(train=SimpleNamespace(epochs=epochs))


def _tiny_dataloader(n: int = 4, in_features: int = 3, out_features: int = 2):
    return [
        (torch.randn(2, in_features), torch.randn(2, out_features)) for _ in range(n)
    ]


def _trivial_loss(outputs, targets):
    return ((outputs - targets) ** 2).mean()


def _build_trainer(tmp_path: Path, epochs: int = 2) -> Trainer:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_trivial_loss,
        cfg=_make_cfg(epochs),
        best_model_path=tmp_path / "best.pt",
    )


def test_fit_runs_and_returns_history(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=2)
    history = trainer.fit(_tiny_dataloader())

    assert isinstance(history, dict)
    assert len(history["train_loss"]) == 2
    assert all(math.isfinite(loss) for loss in history["train_loss"])
    assert math.isfinite(history["best_loss"])


def test_fit_writes_best_checkpoint(tmp_path):
    best_path = tmp_path / "best.pt"
    trainer = _build_trainer(tmp_path, epochs=1)
    trainer.fit(_tiny_dataloader())

    assert best_path.exists()
    state = torch.load(best_path)
    assert "weight" in state and "bias" in state


def test_fit_creates_parent_dir(tmp_path):
    nested = tmp_path / "nested" / "registry" / "best.pt"
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_trivial_loss,
        cfg=_make_cfg(1),
        best_model_path=nested,
    )
    trainer.fit(_tiny_dataloader())
    assert nested.exists()


def test_scheduler_steps_each_epoch(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_trivial_loss,
        cfg=_make_cfg(2),
        scheduler=scheduler,
        best_model_path=tmp_path / "best.pt",
    )
    trainer.fit(_tiny_dataloader())
    # lr halved twice from 0.1 -> 0.025
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.025, rel_tol=1e-6)


def test_validation_loss_tracked(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=1)
    history = trainer.fit(_tiny_dataloader(), val_dataloader=_tiny_dataloader(2))
    assert len(history["val_loss"]) == 1
    assert math.isfinite(history["val_loss"][0])
