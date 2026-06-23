import numpy as np
import pytest
from omegaconf import OmegaConf

from feral_segmentor import constants as C
from feral_segmentor.data.augmentations import (
    BrightnessShift,
    GammaAdjust,
    HorizontalFlip,
    Identity,
    RandomRotate90,
    build_chain,
)


def _aug_cfg(ops):
    return OmegaConf.create({"name": "test", "ops": list(ops)})


# --- build_chain ------------------------------------------------------------
def test_build_chain_empty_ops_is_identity():
    chain = build_chain(_aug_cfg([]))
    assert isinstance(chain, Identity)
    sample = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(chain.augment(sample), sample)


def test_build_chain_variant_label_reflects_order():
    chain = build_chain(_aug_cfg(["HorizontalFlip", "BrightnessShift"]))
    # First op is innermost; variant_label joins inner-first.
    assert chain.variant_label() == "HorizontalFlip_BrightnessShift"


def test_build_chain_applies_inner_first():
    # ops = [HorizontalFlip, BrightnessShift]: flip first, then add brightness.
    chain = build_chain(_aug_cfg(["HorizontalFlip", "BrightnessShift"]))
    sample = np.array([[0.0, 0.2, 0.4]])
    expected = np.clip(np.flip(sample, axis=1) + C.DEFAULT_BRIGHTNESS_SHIFT, 0.0, 1.0)
    np.testing.assert_allclose(chain.augment(sample), expected)


def test_build_chain_unknown_op_raises():
    with pytest.raises(KeyError):
        build_chain(_aug_cfg(["NotARealAug"]))


# --- concrete augmentations (deterministic) ---------------------------------
def test_horizontal_flip_mirrors_width():
    sample = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
    np.testing.assert_array_equal(HorizontalFlip().augment(sample), expected)


def test_random_rotate90_deterministic():
    sample = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.rot90(sample, k=C.DEFAULT_ROTATE90_K, axes=(0, 1))
    np.testing.assert_array_equal(RandomRotate90().augment(sample), expected)


def test_brightness_shift_clips_to_unit_range():
    sample = np.array([[0.0, 0.5, 0.95]])
    out = BrightnessShift(shift=0.1).augment(sample)
    np.testing.assert_allclose(out, np.array([[0.1, 0.6, 1.0]]))


def test_gamma_adjust_applies_power():
    sample = np.array([[0.0, 0.25, 1.0]])
    out = GammaAdjust(gamma=2.0).augment(sample)
    np.testing.assert_allclose(out, np.array([[0.0, 0.0625, 1.0]]))
