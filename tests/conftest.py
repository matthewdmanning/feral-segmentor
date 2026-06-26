from pathlib import Path

import cv2
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_image():
    """Real BGR image loaded from the committed test fixtures."""
    path = FIXTURES_DIR / "american_bulldog_103_original.jpg"
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
