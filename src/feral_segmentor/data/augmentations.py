import difflib
import importlib
from pathlib import Path
from typing import Any

import albumentations as A
import hydra
import numpy as np
from omegaconf import DictConfig

from feral_segmentor.config.store import register_configs
from feral_segmentor.utils import get_logger, to_dtype

log = get_logger(__name__)


def compose_augmentations(cfg: DictConfig) -> A.Compose:
    """Build an Albumentations Compose pipeline from a Hydra AugmentationConfig.

    Parameters
    ----------
    cfg : DictConfig
        An ``AugmentationConfig``-shaped node with an ``ops`` list. Each entry
        must have a ``name`` key and any kwargs accepted by the corresponding
        Albumentations transform.

    Returns
    -------
    A.Compose
        Ready-to-call pipeline; invoke as ``pipeline(image=img)["image"]``.

    Raises
    ------
    ValueError
        If a transform name is not found in the albumentations namespace, with
        fuzzy suggestions sourced from the live module.

    Notes
    -----
    Short names (no dot) are resolved via ``getattr(albumentations, name)``.
    Fully qualified names (containing a dot) are resolved via ``importlib``,
    so any transform reachable by import path is supported without a registry.
    """
    ops = list(cfg.ops) if cfg.ops else []
    transforms = [_instantiate_transform(op) for op in ops]
    return A.Compose(transforms)


def _instantiate_transform(op: Any) -> A.BasicTransform:
    name = op["name"] if hasattr(op, "__getitem__") else op.name
    kwargs = {k: v for k, v in op.items() if k != "name"}

    if "." not in name:
        cls = getattr(A, name, None)
        if cls is None:
            _raise_unknown_transform(name)
        return cls(**kwargs)

    module_path, _, attr = name.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        return cls(**kwargs)
    except (ImportError, AttributeError):
        _raise_unknown_transform(name)


def _raise_unknown_transform(name: str) -> None:
    candidates = [
        k
        for k in dir(A)
        if isinstance(getattr(A, k, None), type)
        and issubclass(getattr(A, k), A.BasicTransform)
    ]
    suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
    msg = f"unknown augmentation {name!r}"
    if suggestions:
        msg += f"; did you mean: {', '.join(suggestions)}?"
    raise ValueError(msg)


def run_augment_stage(cfg: DictConfig) -> None:
    """Apply the configured augmentation pipeline across data/raw -> data/augmented.

    Parameters
    ----------
    cfg : DictConfig
        Top-level Hydra config; must contain ``cfg.augmentation`` and optionally
        ``cfg.data.root`` (defaults to ``"data"``).

    Notes
    -----
    Images are converted to uint8 via ``to_dtype`` before the Albumentations
    pipeline (its expected input format) and written back as uint8. The stage
    is a no-op when the raw directory is absent so DVC never fails in
    environments where data has not been fetched.
    """
    pipeline = compose_augmentations(cfg.augmentation)
    label = cfg.augmentation.name
    log.info("built augmentation pipeline: %s", label)

    root = Path(getattr(cfg.data, "root", "data")) if "data" in cfg else Path("data")
    raw_dir = root / "raw"
    out_dir = root / "augmented"

    if not raw_dir.exists():
        log.info("raw dir %s absent; nothing to augment", raw_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            log.warning("could not read %s; skipping", path)
            continue
        image_u8 = to_dtype(image, np.uint8) if image.dtype != np.uint8 else image
        augmented = pipeline(image=image_u8)["image"]
        dest = out_dir / f"{path.stem}_{label}{path.suffix}"
        cv2.imwrite(str(dest), augmented)
        log.info("wrote %s", dest)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    register_configs()
    run_augment_stage(cfg)


if __name__ == "__main__":
    main()
