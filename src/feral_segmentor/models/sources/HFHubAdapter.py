"""HuggingFace Hub source adapter.

Training workflow  : HFHubAdapter().fetch(cfg) → HFHubAdapter().instantiate(cfg)
Registry workflow  : HFHubAdapter().inspect(cfg)  [via register_model.load_properties]
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, model_info
from omegaconf import DictConfig
from torch import nn

from feral_segmentor.models.ModelProperties import ModelProperties
from feral_segmentor.tasks import CVTask
from feral_segmentor.utils import get_logger

log = get_logger(__name__)

_PIPELINE_TAG_TASKS: dict[str, list[CVTask]] = {
    "image-classification": [CVTask.CLASSIFICATION],
    "object-detection": [CVTask.DETECTION],
    "pose-estimation": [CVTask.POSE],
    "semantic-segmentation": [CVTask.SEG_SEMANTIC],
    "instance-segmentation": [CVTask.SEG_INSTANCE],
    "image-segmentation": [CVTask.SEG_SEMANTIC, CVTask.SEG_INSTANCE],
}


class HFHubAdapter:
    def fetch(self, cfg: DictConfig) -> None:
        """Download cfg.weights.id files to cfg.weights.location. No-op per file if already present."""
        dest = Path(cfg.weights.location)
        dest.mkdir(parents=True, exist_ok=True)
        for filename in cfg.weights.id:
            if (dest / filename).exists():
                log.debug("already cached: %s", dest / filename)
                continue
            log.info("fetching %s from %s", filename, cfg.architecture.id)
            hf_hub_download(
                repo_id=cfg.architecture.id, filename=filename, local_dir=dest
            )

    def instantiate(self, cfg: DictConfig) -> nn.Module:
        """Load model from local weights. Requires fetch() to have run first."""
        import torch

        dest = Path(cfg.weights.location)
        paths = [dest / f for f in cfg.weights.id]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"weight files not found: {missing} — run fetch() first"
            )

        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(str(dest), local_files_only=True)
        except Exception:
            pass

        return torch.load(paths[0], map_location="cpu", weights_only=False)

    def inspect(
        self, cfg: DictConfig, *, fetch_if_needed: bool = False
    ) -> ModelProperties:
        """Return model properties from hub metadata; fetches and loads locally if needed."""
        try:
            info = model_info(cfg.architecture.id)
            tag = getattr(info, "pipeline_tag", None)
            model_cfg = getattr(info, "config", None) or {}
            return ModelProperties(
                n_classes=model_cfg.get("num_labels") or model_cfg.get("num_classes"),
                model_outputs=_PIPELINE_TAG_TASKS.get(tag or "", []),
            )
        except Exception as exc:
            if not fetch_if_needed:
                raise RuntimeError(
                    f"could not inspect {cfg.architecture.id!r} from hub metadata "
                    f"({exc}); pass fetch_if_needed=True to fetch and inspect locally"
                ) from exc

        self.fetch(cfg)
        return _inspect_loaded(self.instantiate(cfg))


def _inspect_loaded(model: nn.Module) -> ModelProperties:
    """Infer n_classes from the last Linear or Conv2d layer of a loaded model."""
    for _, mod in reversed(list(model.named_modules())):
        if isinstance(mod, nn.Linear):
            return ModelProperties(n_classes=mod.out_features)
        if isinstance(mod, nn.Conv2d):
            return ModelProperties(n_classes=mod.out_channels)
    return ModelProperties()
