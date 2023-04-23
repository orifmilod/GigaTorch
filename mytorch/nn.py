import random
from mytorch.engine import Value
from mytorch.visualize import draw_graph, draw_dot

class Neuron:
  def __init__(self, number_of_input, nonlin=True) -> None:
    self.weights = [Value(random.uniform(-1, 1), label='weight') for _ in range(number_of_input)]
    self.bias = Value(random.uniform(-1, 1), label='bias')
    self.nonlin = nonlin

  def __call__(self, x):
    # w * x + b
    activation = sum(wi * Value(xi, label='input') for wi, xi in list(zip(self.weights, x))) + self.bias
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
      result += f"Neuron: {neuron_index} -> {repr(neuron)}"
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
    [2.0],
    # [3.0, -1.0, 0.5],
    # [0.5, 1.0, 1.0],
    # [1.0, 1.0, -1.0]
  ]

  ys = [1.0] #, -1.0, -1.0, 1.0]

  mlp = MLP(1, [2, 1])

  y_pred = [mlp(x) for x in xs]
  print("y_pred", y_pred)

  loss = sum([(y_pred - y_target) ** 2 for y_target, y_pred in zip(ys, y_pred)])
  loss.label = 'loss'
  print("Loss", loss)
  loss.backprop()
  print(mlp)
  # draw_graph(loss)
  draw_dot(loss)
  # draw_graph(mlp)

if __name__ == "__main__":
  main()
