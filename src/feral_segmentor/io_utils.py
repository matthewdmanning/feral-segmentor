import json
from pathlib import Path
from typing import Any


def load_image(path: str | Path):
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def save_image(image, path: str | Path) -> None:
    import cv2

    cv2.imwrite(str(path), image)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(data: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2))
