from functools import reduce
from typing import List
from gigatorch.activation_fn import relu
from gigatorch.loss import cross_entropy_loss, softmax
from gigatorch.nn import MLP
from gigatorch.utils import one_hot
from gigatorch.weight_init import WightInitializer
from gigatorch.tensor import Tensor
from PIL import Image
from abc import ABC, abstractmethod
from os import listdir
from os.path import join
import numpy as np


class Compute(ABC):
    @abstractmethod
    def compute(self, input: Tensor) -> Tensor:
        pass


"""
The MaxPool2D layer extracts the maximum value over the window defined by pool_size
for each dimension along the features axis. The window is shifted by strides in each dimension.

MaxPool2D accepts a 4-dimensional tensor as input. The dimensions represent:
Batch size: The number of samples in a batch. We can do parallel processing if it's more than 1 batch.
Channels: The number of input channels. For example, an RGB image would have 3 channels.
Height: The height of the input.
Width: The width of the input.
"""
class MaxPool2D(Compute):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size


    def compute(self, input: Tensor) -> Tensor:
        assert len(input.shape) == 4, f"can't 2d pool {input.shape}"
        (batch_size, channels, height, width) = input.shape
        assert (height - self.kernel_size) % self.stride == 0, f"Height does not fit the kernel size {self.kernel_size} and stride {self.stride}"
        assert (width - self.kernel_size) % self.stride == 0, f"Width does not fit the kernel size {self.kernel_size} and stride {self.stride}"

        print("Computing maxpool")
        print("Input shape: ", input.shape)

        pooled_height = (height - self.kernel_size) // self.stride + 1
        pooled_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, channels, pooled_height, pooled_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        output[b, c, i, j] = np.max(input.data[b, c, h_start:h_end, w_start:w_end])

        print("\n")
        return Tensor(output)


class Conv2D(Compute):
    """
    (Initial Parameters)
    in_channels: int
    out_channels: int
    kernel_size: int | tuple
    activation_fn: An activation function to apply after computing
    stride: int
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        """
        In a Conv2D layer with 2 input channels and 4 output channels, the number of filters is equal to the number of output channels.
        Each filter in the Conv2D layer is responsible for extracting specific features from the input channels.
        Therefore, in this scenario, there would be 4 filters in the Conv2D layer. Each filter would have a depth of 2 -
        (corresponding to the number of input channels) and produce a single output channel.
        so self.kernes shape is (output_channels, input_channel, kernel_size, kernel_size)
        """
        self.kernels = Tensor(
            [
                [
                    WightInitializer().xavier_uniform(
                        in_channels, out_channels, kernel_size, kernel_size
                    )
                    for _ in range(in_channels)
                ]
                for _ in range(out_channels)
            ]
        )
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn
        self.stride = stride

   def compute(self, input):
        (batch_size, _, height, width) = input.shape
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        output = Tensor(np.zeros((batch_size, self.out_channels, output_height, output_width)))

        for b in range(batch_size):
            for k in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        output[b, k, i, j] = self.activation_fn(
                            np.sum(input[b, :, h_start:h_end, w_start:w_end] * self.kernels[k])
                        )

        return output

class CNN:
    def __init__(self, train_data_dir, test_data_dir, categories):
        print("Initalising CNN class")
        self.categories = categories

        print("Loading training data")
        self.training_data = self._load_data(train_data_dir, categories)
        # print("Loading testing data")
        # self.test_data = self._load_data(test_data_dir, categories)

        self._setup_feature_extraction_layers()
        # TODO: Any better way of computing the input layer?
        # What happens when the input image are of different dimensions
        # input_size = 28
        # for layer in self.features_extraction_layers:
        # input_size = (input_size - layer.kernel_size + 1) / layer.stride

        # For calculating loss we will use the following:
        # loss = L(y, f(s)); where L is the loss function, f is the softmax and s is the output from NN
        # print("NN input layer", input_size)

        self.nn = MLP(5 * 5 * 64, [128, 10], cross_entropy_loss, softmax)

    """
    Feature extraction layer consist of:
    (Convolution + Relu) -> (MaxPooling) -> (Convolution + Relu) -> (MaxPooling)
    """

    def _setup_feature_extraction_layers(self):
        self.features_extraction_layers = [
            Conv2D(1, 32, 3, relu),
            MaxPool2D(2),
            Conv2D(32, 64, 3, relu),
            MaxPool2D(2),
        ]

    def _load_data(self, data_dir, categories):
        data = {}  # {'catergory_name': [datas]}
        for category in categories:
            # Finding all the images under the given dir
            images_path = f"{data_dir}/{category}/"
            images_list = [join(images_path, f) for f in listdir(images_path)]
            # print(f"Number of Images in path {images_path}:", len(images_list))

            # Converting all the images to Tensor type
            for img_path in images_list:
                img_array = Tensor(Image.open(img_path))
                # Since our image is gray scale, we cast it to a 1 dimensional 3d tensor - (1,x,x)
                # For other images, it would be 3 layers (3,x,x)
                img_array = img_array.reshape(1, *img_array.shape)
                if category not in data:
                    data[category] = []

                data[category].append(img_array)

        for key in data:
            print(f"Loaded data for category '{key}':", len(data[key]))

        return data

    """
    Extracts features from the image
    """

    def _extract_features(self, data):
        for layer in self.features_extraction_layers:
            data = layer.compute(data)
        return data

    def _flatten(self, matrix):
        return reduce(lambda x, y: x + y, matrix)

    def train(self):
        # Extract features
        for category in self.training_data:
            for image in self.training_data[category][:2]:
                features = self._extract_features(image)
                features = self._flatten(features)
                print("Features", features.shape)
                # Feed the features to Fully connctec NN
                predictions = self.nn(features)
                print("pred", predictions)
                one_hot_encoding = one_hot(category, [i for i in range(10)])

                loss = self.nn.calc_loss(one_hot_encoding, predictions)
                print("loss", loss)

    def test(self):
        # Extract features
        for category in self.test_data:
            for image in self.test_data[category]:
                features = self._extract_features(image)
                print(f"Extracted features", features)
                # Feed the features to Fully connctec NN
