import random
from mytorch.engine import Value
# from mytorch.visualize import draw_graph, draw_dot

class Neuron:
  def __init__(self, number_of_input, nonlin=True) -> None:
    self.weights = [Value(random.uniform(-1, 1), label="weight") for _ in range(number_of_input)]
    self.bias = Value(random.uniform(-1, 1), label='bias')
    self.nonlin = nonlin

  def __call__(self, x):
    # w * x + b
    activation = sum(wi * xi for wi, xi in list(zip(self.weights, x))) + self.bias
    return activation.tanh()

  def __repr__(self):
    return f"Weights:\n{self.weights} \nBias: \n{self.bias}\n"

class Layer:
  def __init__(self, number_of_inputs, number_of_neurons) -> None:
    self.number_of_inputs = number_of_inputs
    self.number_of_neurons = number_of_neurons

    self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_neurons)]

  def __call__(self, x):
    return [neuron(x) for neuron in self.neurons]

  def __repr__(self):
    result = ''

    neuron_index = 0
    for neuron in self.neurons:
      neuron_index += 1
      result += f"Neuron {neuron_index}:\n{repr(neuron)}"
    return result

class MLP:
  """ Multi Layered Perceptron """

  def __init__(self, number_of_inputs, nuerons_per_layers) -> None:
    layers = [number_of_inputs] + nuerons_per_layers
    self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(nuerons_per_layers))]

  def __call__(self, x):
    for layer in self.layers:
      print("Input x into layer", x)
      x = layer(x)
    return x

  def __repr__(self):
    result = ''

    for i in range(len(self.layers)):
      layer = self.layers[i]
      result += f"Layer {i + 1} with {len(layer.neurons)} neurons accepting {layer.number_of_inputs}\n"
      result += repr(layer)
      result += '###########\n'

    return result

