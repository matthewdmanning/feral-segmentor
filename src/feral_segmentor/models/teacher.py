import torch
import torch.nn as nn

class TeacherModel(nn.Module):
    """
    Placeholder wrapper for YOLOv11x-seg teacher.
    Assumes external pretrained weights are loaded.
    """

    def __init__(self):
        super().__init__()
        self.model = None  # loaded externally

    def load_weights(self, path: str):
        self.model = torch.load(path, map_location="cpu")

    def forward(self, x):
        assert self.model is not None, "Teacher not loaded"
        return self.model(x)
