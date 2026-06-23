from typing import Any

from feral_segmentor.models.base import SegmentationModel


class UnspecifiedModel(SegmentationModel):
    def predict(self, image: Any) -> Any:
        raise NotImplementedError("model architecture not yet chosen")
