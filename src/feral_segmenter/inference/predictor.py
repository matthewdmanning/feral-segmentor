from typing import Any


class Predictor:
    def __init__(self, model: Any, cfg: Any):
        self.model = model
        self.cfg = cfg

    def predict(self, image: Any) -> Any:
        raise NotImplementedError("inference not yet implemented")
