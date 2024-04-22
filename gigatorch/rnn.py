from gigatorch.tensor import Tensor
from gigatorch.nn import Layer
from gigatorch.loss import softmax
import numpy as np
import random


class RNN:
    """
    Recurrent Neural Network
    input_size: Number of inputs (can be one-hot encoded words where lenght of the vector is the size of the vocabulary or can be word embeddings)
    nuerons_per_layers: Number of neurons in each layer
    """

    def __init__(self, input_size, nuerons_per_layers, loss_fn=np.tanh, prob_fn=softmax) -> None:
        layers = [input_size] + nuerons_per_layers
        num_of_layers = len(nuerons_per_layers)
        self.layers = [
            Layer(layers[i], layers[i + 1]) for i in range(num_of_layers)
        ]
        # Add recurrent connection from output to input of the first layer
        self.recurrent_weights = [
            Tensor(random.uniform(-1, 1)) for _ in range(nuerons_per_layers[-1])]

        self.loss_fn = loss_fn
        self.prob_fn = prob_fn

    def __call__(self, input, hidden_state):
        assert hidden_state is not None

        for i, layer in enumerate(self.layers):
            if i == 0:
                input = [input[j] + hidden_state[j] * self.recurrent_weights[j]
                         for j in range(len(input))]
            input = layer(input)
            hidden_state = input  # Output of current layer becomes the hidden_state for the next

        return input, hidden_state

    def calc_loss(self, ys, y_pred):
        # Converting y_pred to probabilities
        loss = sum(self.loss_fn(ys, y_pred), Tensor(0))
        loss.backprop()
        return loss.data

    def __repr__(self):
        result = ""

        for i in range(len(self.layers)):
            layer = self.layers[i]
            result += f"Layer {i + 1} with {len(layer.neurons)} neurons accepting {layer.number_of_inputs} inputs\n"
            result += repr(layer)
            result += "###########\n"

        return result

    def init_hidden_state(self):
        return [Tensor(0) for _ in range(len(self.layers[-1].neurons))]
