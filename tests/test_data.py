import cv2
import numpy as np
import pytest
import torch

from feral_segmentor.data.dataset import SegmentationDataset
from feral_segmentor.data.fetch import fetch_data
from feral_segmentor.data.transforms import preprocess


def test_fetch_data_existing_dir_returns_path(tmp_path):
    result = fetch_data(str(tmp_path))
    assert result == tmp_path.resolve()


def test_fetch_data_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        fetch_data(str(missing))


def test_fetch_data_unsupported_scheme_raises():
    with pytest.raises(ValueError):
        fetch_data("s3://bucket/data")


def test_preprocess_shape_and_range(tiny_image):
    tensor = preprocess(tiny_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == (3, 256, 256)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_grayscale(tiny_image):
    gray = cv2.cvtColor(tiny_image, cv2.COLOR_BGR2GRAY)
    tensor = preprocess(gray)
    assert tuple(tensor.shape) == (3, 256, 256)


def _write_synthetic_dataset(root, count=3, size=32):
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for i in range(count):
        img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (size, size), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"sample_{i}.png"), img)
        cv2.imwrite(str(masks_dir / f"sample_{i}.png"), mask)
    return count


def test_dataset_shapes_and_len(tmp_path):
    count = _write_synthetic_dataset(tmp_path)
    dataset = SegmentationDataset(str(tmp_path))
    assert len(dataset) == count

    image_tensor, mask_tensor = dataset[0]
    assert image_tensor.dtype == torch.float32
    assert tuple(image_tensor.shape) == (3, 256, 256)
    assert mask_tensor.dtype == torch.int64
    assert tuple(mask_tensor.shape) == (256, 256)


def test_dataset_missing_dirs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SegmentationDataset(str(tmp_path))
