from typing import Any


class Trainer:
    def __init__(self, model: Any, cfg: Any):
        self.model = model
        self.cfg = cfg

    def fit(self, dataset: Any) -> None:
        raise NotImplementedError("training loop not yet implemented")
