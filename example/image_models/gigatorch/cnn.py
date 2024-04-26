from functools import reduce
from gigatorch.activation_fn import relu
from gigatorch.loss import cross_entropy_loss, softmax
from gigatorch.nn import MLP
from gigatorch.utils import one_hot
from gigatorch.cnn import Conv2D, MaxPool2D
from gigatorch.tensor import Tensor
from os import listdir
from PIL import Image
from os.path import join
from pathlib import Path



class CNN:
    def __init__(self, train_data_dir, test_data_dir, categories):
        self.categories = categories

        self.training_data = self._load_data(train_data_dir, categories)
        self.test_data = self._load_data(test_data_dir, categories)

        self._setup_feature_extraction_layers()
        # TODO: Any better way of computing the input layer?
        input_shape = (1, 28, 28) 

        # For calculating loss we will use the following:
        # loss = L(y, f(s)); where L is the loss function, f is the softmax and s is the output from NN
        self.nn = MLP(28 * 28, [128, 10], cross_entropy_loss, softmax)

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
        root_path = Path(__file__).absolute().parent
        data = {}  # {'catergory_name': [images]}
        for category in categories:
            # Finding all the images under the given dir
            images_path = Path(f"{root_path}/{data_dir}/{category}/")
            images_list = [join(images_path, f) for f in listdir(images_path)]

            # Converting all the images to Tensor type
            for img_path in images_list:
                img_array = Tensor(Image.open(img_path))
                # Since our image is gray scale, we cast it to a 1 dimensional 3d tensor - (1,x,x)
                # For other images, it would be 3 RGB layers (3,x,x)
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
            for image in self.training_data[category]:
                features = self._extract_features(image)
                features = self._flatten(features)
                print("Features", features.shape)
                # Feed the features to Fully connctec NN
                predictions = self.nn(features)
                print("pred", predictions)
                one_hot_encoding = one_hot(category, [i for i in range(10)])

                loss = self.nn.calc_loss(one_hot_encoding, predictions)
                loss.backprop()
                print("loss", loss)

    def test(self):
        # Extract features
        for category in self.test_data:
            for image in self.test_data[category]:
                features = self._extract_features(image)
                print(f"Extracted features", features)
                # Feed the features to Fully connctec NN


def main():
    cnn = CNN("./mnist/training",
              "./mnist/testing", [i for i in range(1)])


if __name__ == "__main__":
    main()
