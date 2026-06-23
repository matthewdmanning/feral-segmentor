import torch

from feral_segmentor.constants import DEFAULT_IN_CHANNELS, DEFAULT_IMAGE_SIZE
from feral_segmentor.models.base import SegmentationOutput
from feral_segmentor.models.registry import get_model
from feral_segmentor.models.segmentation import StudentSegmenter


def test_get_model_returns_student():
    model = get_model("student")
    assert isinstance(model, StudentSegmenter)


def test_predict_returns_segmentation_output():
    model = get_model("student")
    image = torch.rand(DEFAULT_IN_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    out = model.predict(image)
    assert isinstance(out, SegmentationOutput)
    assert out.mask_logits.shape[-2:] == (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert out.boxes.shape[-1] == 4
