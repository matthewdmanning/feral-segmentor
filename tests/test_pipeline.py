import pytest

from feral_segmenter import pipeline


def test_segment_not_implemented():
    with pytest.raises(NotImplementedError):
        pipeline.segment("image.png")
