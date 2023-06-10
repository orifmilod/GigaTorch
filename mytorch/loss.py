from math import log,  e

from mytorch.engine import Value

def squared_loss(ys, y_pred):
    return sum([(y_pred - y_target) ** 2 for y_target, y_pred in zip(ys, y_pred)])

def binary_cross_entropy_loss(ys, ys_pred):
    return [-log(y_pred) if y == 1 else -log(1 - y_pred) for y, y_pred in zip(ys, ys_pred)]

def softmax(numerator, vector):
    denominator = Value(0)
    temp_e = Value(e)
    for i in vector:
        denominator += (temp_e ** i)
    return temp_e ** numerator / denominator
