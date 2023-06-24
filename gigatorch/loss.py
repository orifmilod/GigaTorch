from gigatorch.tensor import Tensor
import numpy as np


def squared_loss(ys, y_pred):
    return sum([(y_pred - y_target) ** 2 for y_target, y_pred in zip(ys, y_pred)])


def cross_entropy_loss(ys: Tensor, y_pred: Tensor):
    epsilon = 1e-10  # small value to avoid taking log by zero
    ys = Tensor(np.clip(ys.item(), epsilon, 1.0 - epsilon))
    return -(y_pred * ys.log()).sum()


def softmax(logits: Tensor) -> Tensor:
    # Subtracting the logis by it's max to avoid having 'inf' when
    # taking large numbers in power
    logits -= logits.max()
    count = logits.exp()
    return count / count.sum()
