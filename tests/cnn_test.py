from gigatorch.nn import Tensor
from pytest import raises
from gigatorch.cnn import Conv2D, MaxPool2D
from gigatorch.activation_fn import relu
import numpy as np

def test_conv2d_shape_success():
    # Test case 1: Basic functionality
    conv2d = Conv2D(in_channels=3, out_channels=2, kernel_size=2, activation_fn=np.tanh, stride=1)
    input_tensor = Tensor(np.random.rand(1, 3, 4, 4))  # 1 image, 3 channels, 4x4 size
    output = conv2d.compute(input_tensor)
    assert output.shape == (1, 2, 3, 3), "Output shape mismatch in Test Case 1"

    # Test case 2: Stride functionality
    conv2d = Conv2D(in_channels=3, out_channels=2, kernel_size=2, activation_fn=np.tanh, stride=2)
    input_tensor = Tensor(np.random.rand(1, 3, 4, 4))  # 1 image, 3 channels, 4x4 size
    output = conv2d.compute(input_tensor)
    assert output.shape == (1, 2, 2, 2), "Output shape mismatch in Test Case 2"

    # Test case 3: Multiple images
    conv2d = Conv2D(in_channels=3, out_channels=2, kernel_size=2, activation_fn=np.tanh, stride=1)
    input_tensor = Tensor(np.random.rand(2, 3, 4, 4))  # 2 images, 3 channels, 4x4 size
    output = conv2d.compute(input_tensor)
    assert output.shape == (2, 2, 3, 3), "Output shape mismatch in Test Case 3"

def test_conv2d_compute_success():
    # Test case: Check actual values
    conv2d = Conv2D(in_channels=1, out_channels=1, kernel_size=2, activation_fn=lambda x: x, stride=1)
    conv2d.kernels = Tensor(np.array([
        [
            [
                [1, 0],
                [0, 1]
            ]
        ]
    ]))  # Set a fixed kernel
    input_tensor = Tensor(np.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ]))

    output = conv2d.compute(input_tensor)
    expected_output = np.array([[[[6, 8], [12, 14]]]])
    assert np.allclose(output.data, expected_output), "Output values mismatch"


def test_conv2d_kernel_size_larger_than_input():
    conv2d = Conv2D(1, 10, 2, relu)
    conv2d.kernels = Tensor(
        [
            [
                [1, 2],
                [3, 4],
            ]
        ]
    )
    sample_data = Tensor(
        [
            [1],
            [1],
            [1],
        ]
    )

    with raises(Exception):
        conv2d.compute(sample_data)


def test_maxpool2d_success():
    maxpool2d = MaxPool2D(kernel_size=2, stride=1)
    sample_data = Tensor([
        [ # Batch 1
            [  # channel 1
                [1, 2, 3, 4],
                [5, 6, 7, -1],
                [0, 2, 100, 9],
                [0, 0, 0, 0],
            ]
        ]]
    )

    expected = Tensor([
        [ # Batch 1
            [ # Channel 1
                [6, 7, 7],
                [6, 100, 100],
                [2, 100, 100],
            ]
        ]
    ])

    output = maxpool2d.compute(sample_data)
    assert (expected == output).all()

    maxpool2d_with_default_stride = MaxPool2D(kernel_size=2)

    expected = Tensor([
        [ # Batch 1
            [ # Channel 1
                [6, 7],
                [2, 100]
            ]
        ]
    ])

    output = maxpool2d_with_default_stride.compute(sample_data)
    assert (expected == output).all()


def test_maxpool2d_kernel_size_larger_than_input():
    mp = MaxPool2D(2)
    sample_data = [
        [1],
        [1],
        [1],
    ]

    with raises(Exception):
        mp.compute(sample_data)
