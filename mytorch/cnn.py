from .engine import Value
from os import path


class Conv2D:
    """
    (Initial Parameters)
    in_channels: int
    out_channels: int
    kernel_size: int | tuple
    bias: bool
    stride: int | 
    """
    def __init__(in_channels, out_channels, kernel_size, bias, stride=1):
        print()


class CNN:
    def __init__(self, train_data: path, test_data: path):
        print("Init CNN")
        self.train_data = self._load_data(train_data)
        self.test_data = self._load_data(test_data)

    def _load_data(self, file_path: path):
        print("Loading data", file_path)
        # TODO: return dataset loaded in Value() class
        return [Value(0.0)]

    def forward(self):
        pass

    def maxpool(self):
        