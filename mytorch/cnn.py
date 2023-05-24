from .engine import Value
from PIL import Image
import numpy as np


class Conv2D:
    """
    (Initial Parameters)
    in_channels: int
    out_channels: int
    kernel_size: int | tuple
    bias: bool
    stride: int
    """

    def __init__(self, in_channels, kernel):
        pass


class CNN:
    def __init__(self, train_data, test_data):
        self.train_data = self._load_data(train_data)
        self.test_data = self._load_data(test_data)

    def _load_data(self, file_path):
        load_img_rz = np.array(Image.open("kolala.jpeg"))

    def forward(self):
        pass

    def maxpool(self):
        pass

    def train(self, in_channel, num_classes=10):
        pass


def main():
    cnn = CNN("./data/mnist/mnist_train.csv", "./data/mnist/mnist_test.csv")
    cnn.train()
