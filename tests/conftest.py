from pathlib import Path

import cv2
import pytest

OXFORD_PET_DIR = Path(r"C:\GitHub\data\oxford_seg\oxford-iiit-pet")


@pytest.fixture
def tiny_image():
    path = OXFORD_PET_DIR / "images" / "Abyssinian_1.jpg"
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
