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


class Compute(ABC):
    @abstractmethod
    def compute(self, data) -> List[List[Tensor]]:
        pass


class MaxPool2D(Compute):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def compute(self, data_list) -> List[List[Tensor]]:
        print("Computing maxpool")
        print("Size of data", data_list.shape[0])
        print("Number of input", data_list.shape[0])
        output = []
        for data in data_list:
            if len(data) < self.kernel_size or len(data[0]) < self.kernel_size:
                raise Exception("Received data is smaller than the kernel_size")

            new_data = []
            for row_index in range(0, len(data) - self.kernel_size + 1, self.stride):
                row = []
                for column_index in range(
                    0, len(data[row_index]) - self.kernel_size + 1, self.stride
                ):
                    current_max = 0
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            current_max = max(
                                current_max, data[row_index + i][column_index + j]
                            )
                    row.append(current_max)
                new_data.append(row)
            output.append(new_data)
        print("Size of data", output.shape[0])
        print("Number of output", output.shape[0])
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

    def compute(self, data_list):
        output = Tensor([])
        # Iterate for out_channels number of times
        for i in range(self.kernels.shape[0]):
            new_layer = []
            for layer_index in range(data_list.shape[0]):
                data = data_list[layer_index]
                kernel = self.kernels[layer_index]
                print("data", data.shape)
                print("kernel", kernel.shape)

                if data.shape[0] < self.kernel_size or data.shape[1] < self.kernel_size:
                    raise Exception("Received data is smaller than the kernel_size")

                new_data = []
                for row_index in range(
                    0, len(data) - self.kernel_size + 1, self.stride
                ):
                    row = []
                    for column_index in range(data.shape[0] - self.kernel_size + 1):
                        sum = 0
                        for i in range(self.kernel_size):
                            for j in range(self.kernel_size):
                                sum += (
                                    data[row_index + i][column_index + j] * kernel[i][j]
                                )
                        a = self.activation_fn(sum.item())
                        print("item", a, type(a))
                        row.append(a)
                    print("Adding new row", row)
                    new_data.append(row)
                new_layer.append(new_data)
                print("Finishing the layer", new_layer)
            output.append(new_layer)

        print("Outputshape", output.shape)
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
