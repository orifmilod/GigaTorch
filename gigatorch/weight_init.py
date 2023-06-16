from random import uniform
from gigatorch.tensor import Tensor
from math import sqrt


"""
    Wight initaliziation is a technique used to address vanishing gradient problem
    more: https://en.wikipedia.org/wiki/Vanishing_gradient_problem

    Xavier/Gorat weight initializers works well with Sigmoid activation function
    He weight initializers works well with ReLU activation function
"""


class WightInitializer:
    # Using same constants as Keras: https://keras.io/api/layers/initializers/
    def _generate(self, x, y, limit):
        return [[Tensor(uniform(-1, 1) * limit) for _ in range(y)] for _ in range(x)]

    # Paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    def xavier_uniform(self, fan_in: int, fan_out: int, rows: int, columns: int):
        if fan_in == 0 or fan_out == 0:
            raise Exception("fan_in or fan_out cannot be 0")

        limit = sqrt(6.0 / fan_in + fan_out)
        return self._generate(rows, columns, limit)

    def xavier_normal(self, fan_in: int, fan_out: int, rows: int, columns: int):
        if fan_in == 0 or fan_out == 0:
            raise Exception("fan_in or fan_out cannot be 0")

        limit = sqrt(2.0 / fan_in + fan_out)
        return self._generate(rows, columns, limit)

    # Paper: https://arxiv.org/abs/1502.1852
    def he_uniform(self, fan_in: int, rows: int, columns: int):
        if fan_in == 0:
            raise Exception("fan_in or fan_out cannot be 0")

        limit = sqrt(6.0 / fan_in)
        return self._generate(rows, columns, limit)

    def he_normal(self, fan_in, rows: int, columns: int):
        if fan_in == 0:
            raise Exception("fan_in or fan_out cannot be 0")

        limit = sqrt(2.0 / fan_in)
        return self._generate(rows, columns, limit)
