from math import log

def squared_loss(ys, y_pred):
    return sum([(y_pred - y_target) ** 2 for y_target, y_pred in zip(ys, y_pred)])

def binary_cross_entropy_loss(y, y_pred):
    return -log(y_pred) if y == 1 else -log(1 - y_pred)
