"""Architecture registry and model builder.

Maps an architecture name to its ``nn.Module`` class so new student variants can
be added without touching call sites. ``build_model`` constructs a model from a
model :class:`DictConfig` (the ``cfg.model`` node or a bare model DictConfig).
"""

from __future__ import annotations

from omegaconf import DictConfig

from feral_segmentor.models.base import SegmentationModel
from feral_segmentor.models.segmentation import StudentSegmenter
from feral_segmentor.models.teacher import TeacherModel

# Default architecture used when a model config does not name one explicitly.
DEFAULT_ARCH: str = "student"

_ARCHITECTURES: dict[str, type] = {
    "student": StudentSegmenter,
    "teacher": TeacherModel,
}


def build_model(cfg: DictConfig) -> SegmentationModel:
    """Build a segmentation model from a model DictConfig."""
    arch = str(cfg.get("arch", DEFAULT_ARCH))
    try:
        arch_cls = _ARCHITECTURES[arch]
    except KeyError as exc:
        raise KeyError(
            f"unknown architecture {arch!r}; registered: {sorted(_ARCHITECTURES)}"
        ) from exc
    return arch_cls.from_config(cfg)


def get_model(name: str = DEFAULT_ARCH) -> SegmentationModel:
    """Construct a model architecture by name using default arch fields."""
    try:
        arch_cls = _ARCHITECTURES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown architecture {name!r}; registered: {sorted(_ARCHITECTURES)}"
        ) from exc
    return arch_cls()
