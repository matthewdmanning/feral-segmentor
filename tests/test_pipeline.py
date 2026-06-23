import pytest

from feral_segmentor import pipeline


def test_segment_not_implemented():
    with pytest.raises(NotImplementedError):
        pipeline.segment("image.png")
