import pytest

from feral_segmentor.models.registry import get_model


def test_get_model_predict_not_implemented(tiny_image):
    model = get_model("unspecified")
    with pytest.raises(NotImplementedError):
        model.predict(tiny_image)
