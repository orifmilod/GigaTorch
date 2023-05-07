from .nn import MLP
from .visualize import draw_dot, draw_graph

def add_labels(mlp):
  for layer_index in range(len(mlp.layers)):
    layer = mlp.layers[layer_index]
    for neuron_index in range(len(layer.neurons)):
       neuron = mlp.layers[layer_index].neurons[neuron_index]
       for w_index in range(len(neuron.weights)):
        w = mlp.layers[layer_index].neurons[neuron_index].weights[w_index]
        w.label = f"weight for {layer_index}/{neuron_index}/{w_index}"

def loss_fn(ys, y_pred):
  return sum([(y_pred - y_target) ** 2 for y_target, y_pred in zip(ys, y_pred)])

def main():
  xs = [
    [2.0, 3.0],
    # [3.0, -1.0, 0.5],
    # [0.5, 1.0, 1.0],
    # [1.0, 1.0, -1.0],
  ]

  ys = [1.0] #, -1.0, -1.0, 1.0]
  mlp = MLP(2, [2, 1])
  # add_labels(mlp)
  # draw_dot(loss)
  # draw_graph(loss)

  # Training the network
  learning_rate = 0.1
  epochs = 1
  for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    y_pred = [mlp(x) for x in xs][0]
    loss = loss_fn(ys, y_pred)
    loss.backprop()
    print("Loss", loss)
    print(mlp)
    print("Here", mlp.layers[0].neurons[0].weights[0].grad)

    #update the weights and biases
    for layer in mlp.layers:
      for neuron in layer.neurons:
        print(f"Updating neuron w and b {neuron}")

        for w in neuron.weights:
          w.data -= w.grad * learning_rate
        neuron.bias -= neuron.bias.grad * learning_rate
        print(f"Updated neuron w and b {neuron}")

if __name__ == "__main__":
  main()