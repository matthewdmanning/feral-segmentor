import pytest

from feral_segmenter.data.dataset import SegmentationDataset
from feral_segmenter.data.fetch import fetch_data
from feral_segmenter.data.transforms import preprocess


def test_fetch_data_not_implemented():
    with pytest.raises(NotImplementedError):
        fetch_data("source")


def test_preprocess_not_implemented(tiny_image):
    with pytest.raises(NotImplementedError):
        preprocess(tiny_image)


def test_dataset_not_implemented():
    with pytest.raises(NotImplementedError):
        SegmentationDataset("root")
