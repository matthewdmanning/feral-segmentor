"""Model-weight acquisition sources (discriminated by ``cfg.source``).

Each :class:`ModelSource` knows how to materialise a model's weights on disk and
return the produced paths. ``build_model_source`` dispatches on the model
config's ``source`` field.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from omegaconf import DictConfig

from feral_segmentor.models.hub_fetch import pull_model
from feral_segmentor.models.registry import build_model
from feral_segmentor.utils import get_logger

log = get_logger(__name__)


class ModelSource(ABC):
    """Strategy for acquiring a model's weights from a model config."""

    @abstractmethod
    def acquire(self, cfg: DictConfig) -> list[Path]:
        """Materialise weights described by ``cfg`` and return their paths."""
        ...


class HubModelSource(ModelSource):
    """Download weights from the Hugging Face Hub."""

    def acquire(self, cfg: DictConfig) -> list[Path]:
        return pull_model(cfg)


class ScriptModelSource(ModelSource):
    """Run an imported ``module:function`` entrypoint that produces weights."""

    def acquire(self, cfg: DictConfig) -> list[Path]:
        entrypoint = str(cfg.entrypoint)
        if ":" not in entrypoint:
            raise ValueError(
                f"entrypoint must be 'module:function', got {entrypoint!r}"
            )
        module_name, _, func_name = entrypoint.partition(":")
        if not module_name or not func_name:
            raise ValueError(
                f"entrypoint must be 'module:function', got {entrypoint!r}"
            )
        module = importlib.import_module(module_name)
        try:
            func = getattr(module, func_name)
        except AttributeError as exc:
            raise AttributeError(
                f"module {module_name!r} has no attribute {func_name!r}"
            ) from exc
        log.info("Running script entrypoint %s", entrypoint)
        return func(cfg)


class ConfigModelSource(ModelSource):
    """Build a model from config and save its initialised state_dict."""

    def acquire(self, cfg: DictConfig) -> list[Path]:
        model = build_model(cfg)
        weights_dir = Path(cfg.weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        path = weights_dir / f"{cfg.name}.pt"
        torch.save(model.state_dict(), path)
        log.info("Saved config-built model to %s", path)
        return [path]


_SOURCES: dict[str, type[ModelSource]] = {
    "hub": HubModelSource,
    "script": ScriptModelSource,
    "config": ConfigModelSource,
}


def build_model_source(cfg: DictConfig) -> ModelSource:
    """Return the :class:`ModelSource` for ``cfg.source``."""
    source = str(cfg.source)
    try:
        source_cls = _SOURCES[source]
    except KeyError as exc:
        raise KeyError(
            f"unknown model source {source!r}; registered: {sorted(_SOURCES)}"
        ) from exc
    return source_cls()
