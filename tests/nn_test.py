from gigatorch.loss import squared_loss, softmax
from gigatorch.nn import Neuron, Layer, MLP
from gigatorch.tensor import Tensor
from torch import allclose, nn
import torch


def test_neuron():
    number_of_inputs = 3
    neuron = Neuron(number_of_inputs)
    neuron.weights = Tensor([1, 0.5, -1])
    neuron.bias = Tensor(-3)

    x = [2, -1.5, -2.5]
    output = neuron(x)  # (1*2) + (0.5*-1.5) +(-2.5 *-1) + (-3) = 0.75 -> tanh(0.75)

    expected = 0.63514895238
    tol = 1e-6

    assert abs(expected - output.data) < tol


def test_layer():
    number_of_inputs = 3
    neurons_per_layer = 2

    layer = Layer(number_of_inputs, neurons_per_layer)

    neuron_weights = [[1, 0.5, -1], [-1, 4, -3]]  # -> tanh(0.75)  # -> tanh(-0.25)
    neuron_biases = [-3, 2]

    for i in range(neurons_per_layer):
        layer.neurons[i].weights = neuron_weights[i]  # todo: convert to Tensor type
        layer.neurons[i].bias = Tensor(neuron_biases[i])

    x = [2, -1.5, -2.5]
    outputs = layer(x)

    expected = [0.63514895238, 0.90514825364]

    tol = 1e-6
    for i in range(neurons_per_layer):
        assert abs(expected[i] - outputs[i].data) < tol


def test_mlp_forward_pass():
    number_of_inputs = 3  # Input for each layer to the neurons
    neurons_per_layer = [2, 3]  # neurons for each layer
    neuron_weigths = [
        [[-1, 4, -3], [1, 0.5, -1]],  # 1st layer  # -> tanh(-0.25)  # -> tanh(0.75)
        [  # 2nd layer
            [0, 1],  # (tanh(-0.25)* 0) + (tanh(-0.75) * 1) -> tanh(tanh(-0.75))
            [-1, 1],  # (tanh(-0.25)*-1) + (tanh(-0.75) * 1) -> tanh(0.39023028998)
            [-1, 0],  #
        ],
    ]
    neuron_biases = [[-3, 2], [1, -2, -1]]

    mlp = MLP(number_of_inputs, neurons_per_layer, squared_loss, softmax)

    # Setting the weights and biases in my NN
    for i in range(len(neurons_per_layer)):
        for j in range(neurons_per_layer[i]):
            mlp.layers[i].neurons[j].weights = neuron_weigths[i][j]
            mlp.layers[i].neurons[j].bias = Tensor(neuron_biases[i][j])

    x = [2, -1.5, -2.5]
    output = mlp(x)

    model = nn.Sequential(
        nn.Linear(number_of_inputs, neurons_per_layer[0]),
        nn.Tanh(),
        nn.Linear(neurons_per_layer[0], neurons_per_layer[1]),
        nn.Tanh(),
    )

    # Setting the weights and biases in PyTorch
    model[0].weight = nn.Parameter(torch.Tensor(neuron_weigths[0]))
    model[0].bias = nn.Parameter(torch.Tensor(neuron_biases[0]))

    model[2].weight = nn.Parameter(torch.Tensor(neuron_weigths[1]))
    model[2].bias = nn.Parameter(torch.Tensor(neuron_biases[1]))

    expected = model(torch.Tensor(x))

    tol = 1e-6
    for i in range(len(x)):
        assert abs(output[i].data - expected[i].item()) < tol


def test_mlp_with_loss_function():
    number_of_inputs = 3  # Input for each layer to the neurons
    neurons_per_layer = [2, 3]  # neurons for each layer
    neuron_weigths = [
        [[-1, 4, -3], [1, 0.5, -1]],  # 1st layer  # -> tanh(-0.25)  # -> tanh(0.75)
        [  # 2nd layer
            [0, 1],  # (tanh(-0.25) * 0) + (tanh(-0.75) * 1) -> tanh(tanh(-0.75))
            [-1, 1],  # (tanh(-0.25) * -1) + (tanh(-0.75) * 1) -> tanh(0.39023028998)
            [-1, 0],  #
        ],
    ]
    neuron_biases = [[-3, 2], [1, -2, -1]]

    mlp = MLP(number_of_inputs, neurons_per_layer, squared_loss, softmax)

    # Setting the weights and biases in my NN
    for i in range(len(neurons_per_layer)):
        for j in range(neurons_per_layer[i]):
            mlp.layers[i].neurons[j].weights = neuron_weigths[i][j]
            mlp.layers[i].neurons[j].bias = Tensor(neuron_biases[i][j])

    x = [2, -1.5, -2.5]
    output = mlp(x)

    print(output)


def test_mlp_backward_pass():
    number_of_inputs = 3  # Input for each layer to the neurons
    neurons_per_layer = [2, 3, 1]  # neurons for each layer
    neuron_weigths = [
        [[-1, 4, -3], [1, 0.5, -1]],  # 1st layer  # -> tanh(-0.25)  # -> tanh(0.75)
        [  # 2nd layer
            [0, 1],   # (tanh(-0.25) * 0) + (tanh(-0.75) * 1) -> tanh(tanh(-0.75))
            [-1, 1],  # (tanh(-0.25) * -1) + (tanh(-0.75) * 1) -> tanh(0.39023028998)
            [-1, 0],  #
        ],
        [[1, -1, 1]],  # 3rd layer
    ]
    neuron_biases = [[-3, 2], [1, -2, -1], [1]]

    mlp = MLP(number_of_inputs, neurons_per_layer, squared_loss, softmax)

    # Setting the weights and biases in my NN
    for i in range(len(neurons_per_layer)):
        for j in range(neurons_per_layer[i]):
            mlp.layers[i].neurons[j].weights = [Tensor(w) for w in neuron_weigths[i][j]]
            mlp.layers[i].neurons[j].bias = Tensor(neuron_biases[i][j])

    x = [2, -1.5, -2.5]
    x_target = [1]

    result_output = mlp([Tensor(a) for a in x])[0]
    result_loss = mlp.calc_loss([Tensor(a) for a in x_target], [result_output])

    # PyTorch
    model = nn.Sequential(
        nn.Linear(number_of_inputs, neurons_per_layer[0]),
        nn.Tanh(),
        nn.Linear(neurons_per_layer[0], neurons_per_layer[1]),
        nn.Tanh(),
        nn.Linear(neurons_per_layer[1], 1),
        nn.Tanh(),
    )

    # Setting the weights and biases in PyTorch
    model[0].weight = nn.Parameter(torch.Tensor(neuron_weigths[0]))
    model[0].bias = nn.Parameter(torch.Tensor(neuron_biases[0]))

    model[2].weight = nn.Parameter(torch.Tensor(neuron_weigths[1]))
    model[2].bias = nn.Parameter(torch.Tensor(neuron_biases[1]))

    model[4].weight = nn.Parameter(torch.Tensor(neuron_weigths[2]))
    model[4].bias = nn.Parameter(torch.Tensor(neuron_biases[2]))

    expected_output = model(torch.Tensor(x))
    mse_loss = nn.MSELoss()

    expected_loss = mse_loss(expected_output, torch.Tensor(x_target))
    expected_loss.backward()

    tol = 1e-6
    assert abs(expected_loss.item() - result_loss) < tol
    assert abs(expected_output.item() - result_output.data) < tol

    for layer_index in range(len(neurons_per_layer)):
        py_layer = model[layer_index * 2].weight.grad
        my_layer = mlp.layers[layer_index]
        for neuron_index in range(neurons_per_layer[layer_index]):
            neuron = my_layer.neurons[neuron_index]
            ws = [w.grad for w in neuron.weights]
            assert allclose(Tensor(ws), py_layer[neuron_index], atol=tol)
