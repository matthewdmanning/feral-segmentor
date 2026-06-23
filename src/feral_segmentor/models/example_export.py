"""Example script entrypoint backing ``conf/model/script.yaml``.

Builds a :class:`StudentSegmenter` from the model config's architecture fields
and saves its state_dict, returning the produced path.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from feral_segmentor.models.registry import build_model
from feral_segmentor.utils import get_logger

log = get_logger(__name__)


def export(cfg: DictConfig) -> list[Path]:
    """Build a student model from config and save its weights."""
    model = build_model(cfg)
    weights_dir = Path(cfg.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / f"{cfg.name}.pt"
    torch.save(model.state_dict(), path)
    log.info("Exported model to %s", path)
    return [path]
