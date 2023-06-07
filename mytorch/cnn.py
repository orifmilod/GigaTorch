from functools import reduce
from typing import List
from mytorch.activation_fn import relu
from mytorch.loss import binary_cross_entropy_loss
from mytorch.nn import MLP
from mytorch.weight_init import WightInitializer
from .engine import Value
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from os import listdir
from os.path import join


class Compute(ABC):
    @abstractmethod
    def compute(self, data) -> List[List[Value]]:
        pass


class MaxPool2D(Compute):
    def __init__(self, in_channels, out_channels, kernel_size, stride = None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def compute(self, data) -> List[List[Value]]:
        if len(data) < self.kernel_size or len(data[0]) < self.kernel_size:
            raise Exception("Received data is smaller than the kernel_size")

        # print("Computing MaxPool2d layer")
        new_data = []
        for row_index in range(0, len(data) - self.kernel_size + 1, self.stride):
            row = []
            for column_index in range(0, len(data[row_index]) - self.kernel_size + 1, self.stride):
                current_max = Value(0)
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        current_max = max(
                            current_max, data[row_index + i][column_index + j]
                        )
                row.append(current_max)
            new_data.append(row)

        return new_data


class Conv2D(Compute):
    """
    (Initial Parameters)
    in_channels: int
    out_channels: int
    kernel_size: int | tuple
    activation_fn: An activation function to apply after computing
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = WightInitializer().xavier_uniform(in_channels, out_channels, 3, 3)
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn
        self.stride = stride

    def compute(self, data) -> List[List[Value]]:
        if len(data) < self.kernel_size or len(data[0]) < self.kernel_size:
            raise Exception("Received data is smaller than the kernel_size")

        # print("Computing Conv2D layer")
        new_data = []
        for row_index in range(0, len(data) - self.kernel_size + 1, self.stride):
            row = []
            for column_index in range(len(data[0]) - self.kernel_size + 1):
                sum = Value(0)
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        sum += data[row_index + i][column_index + j] * self.kernel[i][j]
                row.append(self.activation_fn(sum))
            new_data.append(row)

        return new_data


class CNN:
    def __init__(self, train_data_dir, test_data_dir, categories):
        print("Initalising CNN class")

        print("Loading training data")
        self.training_data = self._load_data(train_data_dir, categories)
        print("Loading testing data")
        self.test_data = self._load_data(test_data_dir, categories)

        self._setup_feature_extraction_layers()
        # TODO: Any better way of computing the input layer?
        # What happens when the input image are of different dimensions
        input_layer = 0
        for layer in self.features_extraction_layers:
            input_layer += layer.kernel_size

        self.nn = MLP(input_layer, [16, 32, 10], binary_cross_entropy_loss)

    def _load_data(self, data_dir, categories):
        data = {}  # {'catergory_name': [datas]}
        for category in categories:
            # Finding all the images under the given dir
            images_path = f"{data_dir}/{category}/"
            images_list = [join(images_path, f) for f in listdir(images_path)]
            # print(f"Number of Images in path {images_path}:", len(images_list))

            # Converting all the images to Value type
            for img_path in images_list:
                img_array = np.array(Image.open(img_path))
                converted_image = []

                for row_index in range(img_array.shape[0]):
                    row = []
                    for column_index in range(img_array.shape[1]):
                        row.append(Value(img_array[row_index][column_index]))
                    converted_image.append(row)

                if category not in data:
                    data[category] = []

                data[category].append(converted_image)

        for key in data:
            print(f"Loaded data for category '{key}':", len(data[key]))

        return data

    """
    Feature extraction layer consist of:
    (Convolution + Relu) -> (MaxPooling) -> (Convolution + Relu) -> (MaxPooling)
    """

    def _setup_feature_extraction_layers(self):
        self.features_extraction_layers = [
            Conv2D(1, 32, 3, relu),
            MaxPool2D(1, 1, 2),
            Conv2D(32, 64, 3, relu),
            MaxPool2D(1, 1, 2),
        ]

    """
    Extracts features from the image
    """

    def _extract_features(self, data):
        for layer in self.features_extraction_layers:
            data = layer.compute(data)
        return data

    def _flatten(self, matrix):
        return reduce(lambda x,y :x+y , matrix)

    def train(self):
        # Extract features
        for category in self.training_data:
            for image in self.training_data[category][:1]:
                features = self._extract_features(image)
                features = self._flatten(features)
                print("Features", features)
                predictions = self.nn(features)
                print("Pred", predictions)
                self.nn.calc_loss([1 if i is category else 0 for i in range(len(self.training_data))], predictions)

                # Feed the features to Fully connctec NN

    def test(self):
        # Extract features
        for category in self.test_data:
            for image in self.test_data[category]:
                features = self._extract_features(image)
                print(f"Extracted features", features)
                # Feed the features to Fully connctec NN
