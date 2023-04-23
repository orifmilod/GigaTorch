from mytorch.nn import Neuron, Layer, MLP
from mytorch.engine import Value
from torch import nn

def test_neuron():
  number_of_inputs = 3
  neuron = Neuron(number_of_inputs)
  neuron.weights = [Value(1), Value(0.5), Value(-1)]
  neuron.bias = Value(-3)

  x = [2, -1.5, -2.5]
  output = neuron(x) # (1*2) + (0.5*-1.5) +(-2.5 *-1) + (-3) = 0.75 -> tanh(0.75)

  expected = 0.63514895238
  tol = 1e-6

  assert abs(expected - output.data) < tol


def test_layer():
  number_of_inputs = 3
  number_of_neurons = 2

  layer = Layer(number_of_inputs, number_of_neurons)

  neuron_weights = [
    [1, 0.5, -1], # -> tanh(0.75)
    [-1, 4, -3] # -> tanh(-0.25)
  ]
  neuron_biases = [-3, 2]

  for i in range(number_of_neurons):
    layer.neurons[i].weights = neuron_weights[i] # todo: convert to Value type
    layer.neurons[i].bias = Value(neuron_biases[i])

  x = [2, -1.5, -2.5]
  outputs = layer(x)

  expected = [
    0.63514895238,
    0.90514825364
  ]

  tol = 1e-6
  for i in range(number_of_neurons):
    assert abs(expected[i] - outputs[i].data) < tol

#TODO: Write this test comparing to pytorch
def test_mlp():
  number_of_inputs  = 3 # Input for each layer to the neurons
  number_of_neurons = [2, 3] # neurons for each layer
  number_of_layers = 2
  neuron_weigths = [
    [ # 1st layer
      [-1, 4, -3],  # -> tanh(-0.25)
      [1, 0.5, -1]  # -> tanh(0.75)
    ],
    [ # 2nd layer
      [0, 1],  # (tanh(-0.25)* 0) + (tanh(-0.75) * 1) -> tanh(tanh(-0.75))
      [-1, 1], # (tanh(-0.25)*-1) + (tanh(-0.75) * 1) -> tanh(0.39023028998)
      [-1, 0]  # 
    ]
  ]
  neuron_biases = [
    [-3, 2],
    [1, -2, -1]
  ]

  mlp = MLP(number_of_inputs, number_of_neurons)

  for i in range(number_of_layers):
    for j in range(number_of_neurons[i]):
      mlp.layers[i].neurons[j].weights = neuron_weigths[i][j]
      mlp.layers[i].neurons[j].bias = Value(neuron_biases[i][j])

  x = [2, -1.5, -2.5]
  print("Start nn", x)
  output = mlp(x)
  print(output)

  model = nn.Sequential(
    nn.Linear(number_of_inputs, number_of_neurons[0]),
    nn.Linear(number_of_neurons[0], number_of_neurons[1]),
  )
  expected = [
    -0.56158748006,
    0.37155874194
  ]

  assert False


def test_mlp_with_loss_function():
  pass
