import abc
from pathlib import Path
from typing import Any, Callable

import hydra
from omegaconf import DictConfig

from feral_segmenter.utils import get_logger

log = get_logger(__name__)


class Augmentation(abc.ABC):
    """Composes via constructor chaining: Outer(Inner()).augment(x) == Outer._apply(Inner._apply(x))."""

    def __init__(self, inner: "Augmentation | None" = None):
        self.inner = inner

    def augment(self, sample: Any) -> Any:
        if self.inner is not None:
            sample = self.inner.augment(sample)
        return self._apply(sample)

    @abc.abstractmethod
    def _apply(self, sample: Any) -> Any: ...

    def _own_params(self) -> dict[str, str | float]:
        return {}

    def to_params(self, index: int = 0) -> dict[str, str | float]:
        flat = {f"{index}.{type(self).__name__}.{k}": v for k, v in self._own_params().items()}
        if self.inner is not None:
            flat |= self.inner.to_params(index=index + 1)
        return flat

    def variant_label(self) -> str:
        names = [type(self).__name__]
        if self.inner is not None:
            names.insert(0, self.inner.variant_label())
        return "_".join(names)


class FunctionAugmentation(Augmentation):
    """Adapter wrapping an albumentations/torchvision-style transform callable."""

    def __init__(self, fn: Callable, inner: "Augmentation | None" = None, **kwargs):
        super().__init__(inner)
        self.fn = fn
        self.kwargs = kwargs

    def _apply(self, sample: Any) -> Any:
        raise NotImplementedError("backend call convention not yet chosen (albumentations vs torchvision)")

    def _own_params(self) -> dict[str, str | float]:
        return {"fn": type(self.fn).__name__, **self.kwargs}


def run_augment_stage(cfg: DictConfig) -> None:
    raise NotImplementedError("augmentation chain construction from config not yet chosen")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_augment_stage(cfg)


if __name__ == "__main__":
    main()
