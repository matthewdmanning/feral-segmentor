import tempfile
from pathlib import Path

from feral_segmentor.io_utils import load_image, read_json, save_image, write_json


def test_read_write_json_roundtrip():
    data = {"a": 1, "b": [1, 2, 3]}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.json"
        write_json(data, path)
        assert read_json(path) == data


def test_load_save_image_roundtrip(tiny_image):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.png"
        save_image(tiny_image, path)
        loaded = load_image(path)
        assert loaded.shape == tiny_image.shape
