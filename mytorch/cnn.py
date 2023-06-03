from mytorch import weight_init
from .engine import Value
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from os import listdir
from os.path import join
from mytorch.weight_init import WightInitializer


class Compute(ABC):
    @abstractmethod
    def compute(self, data):
        pass


class MaxPool2D(Compute):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def compute(self, data):
        if len(data) < self.kernel_size or len(data[0]) < self.kernel_size:
            raise Exception("Received data is smaller than the kernel_size")

        print("Computing MaxPool2d layer")
        new_data = []
        for row_index in range(len(data) - self.kernel_size + 1):
            row = []
            for column_index in range(len(data[0]) - self.kernel_size + 1):
                current_max = 0
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
    bias: bool
    stride: int
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = WightInitializer().xavier_normal(3, 3)
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn

    def compute(self, data):
        if len(data) < self.kernel_size or len(data[0]) < self.kernel_size:
            raise Exception("Received data is smaller than the kernel_size")

        print("Computing Conv2D layer")
        new_data = []
        for row_index in range(len(data) - self.kernel_size + 1):
            row = []
            for column_index in range(len(data[0]) - self.kernel_size + 1):
                sum = 0
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        sum += data[row_index + i][column_index + j] * self.kernel[i][j]
                row.append(sum)
            new_data.append(row)

        return new_data


class CNN:
    def __init__(self, train_data_dir, test_data_dir, categories):
        print("Initalising CNN class")

        print("Loading training data")
        self.training_data = self._load_data(train_data_dir, categories)
        print("Loading testing data")
        self.training_data = self._load_data(test_data_dir, categories)

        self._setup_feature_extraction_layers()

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
        # TODO: Extract this to list of activation_fn list
        def relu(x):
            return max(0, x)

        self.features_extraction_layers = [
            Conv2D(1, 20, 3, relu),
            MaxPool2D(20, 16, 3),
            Conv2D(16, 12, 3, relu),
            MaxPool2D(20, 16, 3),
        ]

    """
    Extracts features from the image
    """

    def _extract_features(self, data):
        for layer in self.features_extraction_layers:
            data = layer.compute(data)
        return data

    def train(self):
        # Extract features
        for category in self.training_data:
            for image in self.training_data[category]:
                features = self._extract_features(image)
                print(f"Extracted features", features)
                # TODO: Feed the features to Fully connctec NN

    def test(self):
        pass
