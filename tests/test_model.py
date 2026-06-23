import pytest
import torch
from hydra import compose, initialize

from feral_segmentor.config.store import register_configs
from feral_segmentor.models.base import SegmentationOutput
from feral_segmentor.models.registry import build_model


@pytest.fixture(autouse=True)
def _registered():
    register_configs()


def _compose(overrides=None):
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name="config", overrides=overrides or [])


def test_forward_logits_shape():
    cfg = _compose()
    model = build_model(cfg.model)

    b, h, w = 2, 64, 64
    x = torch.rand(b, cfg.model.in_channels, h, w)
    logits = model(x)
    assert logits.shape == (b, cfg.model.num_classes, h, w)


def test_predict_returns_segmentation_output():
    cfg = _compose()
    model = build_model(cfg.model)

    image = torch.rand(cfg.model.in_channels, 48, 48)
    out = model.predict(image)

    assert isinstance(out, SegmentationOutput)
    assert out.mask_logits.shape == (cfg.model.num_classes, 48, 48)
    assert out.boxes.shape[-1] == 4
    assert out.scores.shape[0] == out.boxes.shape[0]
    assert out.labels.shape[0] == out.boxes.shape[0]


def test_forward_odd_input_size():
    cfg = _compose()
    model = build_model(cfg.model)
    x = torch.rand(1, cfg.model.in_channels, 50, 70)
    logits = model(x)
    assert logits.shape[-2:] == (50, 70)
