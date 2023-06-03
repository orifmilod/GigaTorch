from .nn import MLP
from mytorch.engine import Value
from .cnn import CNN


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
        [Value(2.0), Value(3.0), Value(-1.5)],
        [Value(3.0), Value(-1.0), Value(0.5)],
        [Value(0.5), Value(1.0), Value(1.0)],
        [Value(1.0), Value(1.0), Value(-1.0)],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]
    mlp = MLP(3, [4, 4, 1])

    # Training the network
    learning_rate = 0.01
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        y_pred = [mlp(x) for x in xs][0]
        loss = mlp.calc_loss(ys, y_pred)
        print("Loss", 100 - loss.data)

        # update the weights and biases
        for layer in mlp.layers:
            for neuron in layer.neurons:
                for w in neuron.weights:
                    w.data -= w.grad * learning_rate
                neuron.bias -= neuron.bias.grad * learning_rate


if __name__ == "__main__":
    # main()
    cnn = CNN("./data/mnist/training", "./data/mnist/testing", [i for i in range(10)])
    cnn.train()
    cnn.test()
