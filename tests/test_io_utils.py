"""Contract tests for the plain disk I/O helpers in feral_segmentor.io_utils.

DatasetSource (also defined in this module) already has thorough fixture-backed
coverage in tests/test_dataset_loading.py — not duplicated here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feral_segmentor.io_utils import load_image, read_json, save_image, write_json

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_bgr_image(size: tuple[int, int] = (16, 24)) -> np.ndarray:
    """Deterministic synthetic BGR image of shape (H, W, 3), dtype uint8."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (*size, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# load_image / save_image — roundtrip contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1, 1), (16, 24), (64, 64)])
def test_save_then_load_image_roundtrips_shape_and_values(tmp_path, size):
    image = _synthetic_bgr_image(size)
    path = tmp_path / "roundtrip.png"

    save_image(image, path)
    loaded = load_image(path)

    assert loaded.shape == image.shape
    np.testing.assert_array_equal(loaded, image)


def test_save_then_load_real_fixture_image_roundtrips(tmp_path, tiny_image):
    path = tmp_path / "roundtrip.png"

    save_image(tiny_image, path)
    loaded = load_image(path)

    assert loaded.shape == tiny_image.shape
    np.testing.assert_array_equal(loaded, tiny_image)


def test_load_image_missing_file_returns_none(tmp_path):
    assert load_image(tmp_path / "missing.png") is None


# ---------------------------------------------------------------------------
# read_json / write_json — roundtrip contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"a": 1, "b": [1, 2, 3]},
        {"nested": {"x": 1.5, "y": [True, False, None]}},
        [1, "two", 3.0],
    ],
)
def test_write_then_read_json_roundtrips(tmp_path, data):
    path = tmp_path / "data.json"

    write_json(data, path)
    result = read_json(path)

    assert result == data


def test_write_json_pretty_prints_with_indentation(tmp_path):
    path = tmp_path / "data.json"

    write_json({"a": 1}, path)

    text = path.read_text()
    assert "\n" in text
    assert "  " in text


def test_read_json_accepts_str_path(tmp_path):
    path = tmp_path / "data.json"
    write_json({"a": 1}, path)

    assert read_json(str(path)) == {"a": 1}
