import random
import engine

class Neuron:
  def __init__(self, number_of_neurons) -> None:
    self.weights = [engine.Value(random.uniform(-1, 1)) for _ in range(number_of_neurons)]
    self.bias = engine.Value(random.uniform(-1, 1))

  def __call__(self, x):
    # w * x + b
    activation = sum(wi * xi for wi, xi in list(zip(self.weights, x))) + self.bias
    return activation.tanh()


class Layer:
  def __init__(self, number_of_inputs, number_of_neurons) -> None:
    self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_neurons)]

  def __call__(self, x):
    return [neuron(x) for neuron in self.neurons]


class MLP:
  """ Multi Layered Perceptron """

  def __init__(self, number_of_inputs, list_of_output) -> None:
    layers = [number_of_inputs] + list_of_output
    self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

x = [2.0, 3.0, -1.0]
mlp = MLP(3, [4, 4, 1])
print(mlp(x))
