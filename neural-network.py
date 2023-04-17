import random
from mytorch.engine import Value
import uuid
from mytorch.visualize import draw_graph

counter = 0

class Neuron:
  def __init__(self, number_of_neurons, nonlin=True) -> None:
    self.weights = [Value(random.uniform(-1, 1)) for _ in range(number_of_neurons)]
    self.bias = Value(random.uniform(-1, 1))
    self.nonlin = nonlin

  def __call__(self, x):
    # w * x + b
    activation = sum(wi * xi for wi, xi in list(zip(self.weights, x))) + self.bias
    activation.label = uuid.uuid1()
    return activation.tanh()

  def __repr__(self):
    return f"Weights:\n{self.weights} \nBias: \n{self.bias}\n"

class Layer:
  def __init__(self, number_of_inputs, number_of_neurons) -> None:
    self.number_of_inputs = number_of_inputs
    self.number_of_neurons = number_of_neurons

    self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_neurons)]

  def __call__(self, x):
    output = [neuron(x) for neuron in self.neurons]
    return output[0] if len(output) == 1 else output

  def __repr__(self):
    result = ''
    for neuron in self.neurons:
      result += repr(neuron)
    return result

class MLP:
  """ Multi Layered Perceptron """

  def __init__(self, number_of_inputs, list_of_output) -> None:
    layers = [number_of_inputs] + list_of_output
    self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(list_of_output))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def __repr__(self):
    result = ''

    for i in range(len(self.layers)):
      layer = self.layers[i]
      result += f"Layer {i + 1} with {len(layer.neurons)} neurons accepting {layer.number_of_inputs}\n"
      result += repr(layer)
      result += '\n'

    return result

def main():
  xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
  ]

  ys = [1.0, -1.0, -1.0, 1.0]

  mlp = MLP(3, [4, 4, 1])
  # print(mlp)
  y_pred = [mlp(x) for x in xs]
  # print("Predicted", y_pred)

  loss = sum([(y_pred - y_target) ** 2 for y_target, y_pred in zip(ys, y_pred)])
  print("Loss", loss)
  loss.backprop()
  draw_graph(loss)
  # print(mlp)

if __name__ == "__main__":
  main()
