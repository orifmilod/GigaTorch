from pytest import raises
from mytorch.cnn import Conv2D, MaxPool2D
from mytorch.activation_fn import relu


def test_conv2d_success():
    conv2d = Conv2D(1, 10, 2, relu)
    conv2d.kernel = [
        [1, 2],
        [3, 4],
    ]
    sample_data = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]

    output = conv2d.compute(sample_data)
    # Output size is (m-n+1)
    assert len(output) == len(sample_data) - len(conv2d.kernel) + 1
    assert len(output) == len(sample_data[0]) - len(conv2d.kernel) + 1

    expected = [[10, 10], [10, 10]]

    assert output == expected


def test_conv2d_kernel_size_larger_than_input():
    conv2d = Conv2D(1, 10, 2, relu)
    conv2d.kernel = [
        [1, 2],
        [3, 4],
    ]
    sample_data = [
        [1],
        [1],
        [1],
    ]

    with raises(Exception):
        conv2d.compute(sample_data)


def test_maxpool2d_success():
    maxpool2d = MaxPool2D(1, 1, 2, 1)
    sample_data = [
        [1, 2, 3, 4],
        [5, 6, 7, -1],
        [0, 2, 100, 9],
        [0, 0, 0, 0],
    ]

    expected = [
        [6, 7, 7],
        [6, 100, 100],
        [2, 100, 100],
    ]

    output = maxpool2d.compute(sample_data)
    assert expected == output

    maxpool2d_with_default_stride = MaxPool2D(1, 1, 2)

    expected = [
        [6, 7],
        [2, 100],
    ]

    output = maxpool2d_with_default_stride.compute(sample_data)
    assert expected == output


def test_maxpool2d_kernel_size_larger_than_input():
    mp = MaxPool2D(1, 1, 2)
    sample_data = [
        [1],
        [1],
        [1],
    ]

    with raises(Exception):
        mp.compute(sample_data)
