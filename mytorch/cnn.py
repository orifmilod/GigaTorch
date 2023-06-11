from functools import reduce
from typing import List
from mytorch.activation_fn import relu
from mytorch.loss import binary_cross_entropy_loss, softmax
from mytorch.nn import MLP
from mytorch.utils import one_hot
from mytorch.weight_init import WightInitializer
from mytorch.tensor import Tensor
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from os import listdir
from os.path import join


class Compute(ABC):
    @abstractmethod
    def compute(self, data) -> List[List[Tensor]]:
        pass

class MaxPool2D(Compute):
    def __init__(self, kernel_size, stride = None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def compute(self, data_list) -> List[List[Tensor]]:
        print("Computing maxpool")
        print("Size of data", len(data_list[0]))
        print("Number of input", len(data_list))
        output = []
        for data in data_list:
            if len(data) < self.kernel_size or len(data[0]) < self.kernel_size:
                raise Exception("Received data is smaller than the kernel_size")

            new_data = []
            for row_index in range(0, len(data) - self.kernel_size + 1, self.stride):
                row = []
                for column_index in range(0, len(data[row_index]) - self.kernel_size + 1, self.stride):
                    current_max = Tensor(0)
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            current_max = max(
                                current_max, data[row_index + i][column_index + j]
                            )
                    row.append(current_max)
                new_data.append(row)
            output.append(new_data)
        print("Size of data", len(output[0]))
        print("Number of output", len(output))
        print("\n")
        return output


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
        # In a Conv2D layer, the number of filters corresponds to the number of output channels. 
        # if the input_channel is 3 and the output_channel is 5, then there would be 5 filters in the Conv2D layer. 
        # Each filter will have a depth of 3 (matching the input_channel), and the spatial dimensions of 
        # the filters will be determined by the kernel size specified in the Conv2D layer.
        self.kernels = Tensor([
            [
                WightInitializer().xavier_uniform(in_channels, out_channels, kernel_size, kernel_size)
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ])

        self.kernel_size = kernel_size
        self.activation_fn = activation_fn
        self.stride = stride

        print("Conv2D with", in_channels, out_channels, kernel_size, "has", self.kernels.shape)

    def compute(self, data_list):
        print("computing conv2d")
        print("Size of data", len(data_list[0]))
        print("Number of input", len(data_list))
        output = []
        print("Number of kernels", len(self.kernels))
        for kernel in self.kernels:
            output.append([])
            for data in data_list:
                if len(data) < self.kernel_size or len(data[0]) < self.kernel_size:
                    raise Exception("Received data is smaller than the kernel_size")

                new_data = []
                for row_index in range(0, len(data) - self.kernel_size + 1, self.stride):
                    row = []
                    for column_index in range(len(data[0]) - self.kernel_size + 1):
                        sum = Tensor(0)
                        for i in range(self.kernel_size):
                            for j in range(self.kernel_size):
                                sum += data[row_index + i][column_index + j] * kernel[i][j]
                        row.append(self.activation_fn(sum))
                    new_data.append(row)
                output[1].append(new_data)
        print("Size of data", len(output[0]))
        print("Number of output", len(output))
        print("\n")
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
        input_size = 28
        for layer in self.features_extraction_layers:
            input_size = (input_size - layer.kernel_size + 1) / layer.stride

        # For calculating loss we will use the following:
        # loss = L(y, f(s)); where L is the loss function, f is the softmax and s is the output from NN
        print("NN input layer", input_size)

        self.nn = MLP(5 * 5 * 64, [128, 10], binary_cross_entropy_loss, softmax)

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
                img_array = np.array(Image.open(img_path))
                converted_image = []

                for row_index in range(img_array.shape[0]):
                    row = []
                    for column_index in range(img_array.shape[1]):
                        row.append(Tensor(img_array[row_index][column_index]))
                    converted_image.append(row)

                if category not in data:
                    data[category] = []

                data[category].append(converted_image)

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
        return reduce(lambda x, y :x+y, matrix)

    def train(self):
        # Extract features
        for category in self.training_data:
            for image in self.training_data[category][:2]:
                features = self._extract_features([image])
                features = self._flatten(features)
                print("Features", len(features))
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
