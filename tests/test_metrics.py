import pytest

from feral_segmenter.training.metrics import dice_score, mean_iou


def test_mean_iou_not_implemented():
    with pytest.raises(NotImplementedError):
        mean_iou(None, None)


def test_dice_score_not_implemented():
    with pytest.raises(NotImplementedError):
        dice_score(None, None)
