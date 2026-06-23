from typing import Any

from feral_segmentor.data.fetch import fetch_data
from feral_segmentor.data.transforms import preprocess
from feral_segmentor.io_utils import load_image
from feral_segmentor.models.registry import get_model


def segment(image_path: str, model_name: str = "unspecified") -> Any:
    path = fetch_data(image_path)
    image = load_image(path)
    preprocessed = preprocess(image)
    model = get_model(model_name)
    return model.predict(preprocessed)
