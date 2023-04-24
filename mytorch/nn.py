import random
from .engine import Value
from .visualize import draw_graph, draw_dot

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
      result += f"Neuron: {neuron_index}\n{repr(neuron)}"
    return result

class MLP:
  """ Multi Layered Perceptron """

  def __init__(self, number_of_inputs, nuerons_per_layers) -> None:
    layers = [number_of_inputs] + nuerons_per_layers
    self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(nuerons_per_layers))]

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

def add_labels(mlp):
  for layer_index in range(len(mlp.layers)):
    layer = mlp.layers[layer_index]
    for neuron_index in range(len(layer.neurons)):
       neuron = mlp.layers[layer_index].neurons[neuron_index]
       for w_index in range(len(neuron.weights)):
        w = mlp.layers[layer_index].neurons[neuron_index].weights[w_index]
        w.label = f"weight for {layer_index}/{neuron_index}/{w_index}"

def main():
  xs = [
    [2.0, 3.0],
    # [3.0, -1.0, 0.5],
    # [0.5, 1.0, 1.0],
    # [1.0, 1.0, -1.0],
  ]

  ys = [1.0] #, -1.0, -1.0, 1.0]

  mlp = MLP(2, [2, 1])

  y_pred = [mlp(x) for x in xs]
  loss = sum([(y_pred - y_target) ** 2 for y_target, y_pred in zip(ys, y_pred)])
  loss.backprop()
  print("Loss", loss)
  print(mlp)
  add_labels(mlp)
  # draw_graph(loss)
  draw_dot(loss)
  # draw_graph(mlp)

if __name__ == "__main__":
  main()
