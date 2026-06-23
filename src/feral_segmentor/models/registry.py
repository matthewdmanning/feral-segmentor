from feral_segmentor.models.base import SegmentationModel
from feral_segmentor.models.segmentation import UnspecifiedModel

_MODELS: dict[str, type[SegmentationModel]] = {
    "unspecified": UnspecifiedModel,
}


def get_model(name: str) -> SegmentationModel:
    return _MODELS[name]()
