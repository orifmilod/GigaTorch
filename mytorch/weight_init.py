from random import uniform
from mytorch.engine import Value
from math import sqrt


class WightInitializer:
    # Using same constants as Keras: https://keras.io/api/layers/initializers/
    def _generate(self, x, y, limit):
        return [[Value(uniform(-1, 1) * limit) for _ in range(x)] for _ in range(y)]

    def xavier_uniform(self, fan_in: int, fan_out: int):
        limit = sqrt(6.0 / fan_in + fan_out)
        return self._generate(fan_in, fan_out, limit)

    def xavier_normal(self, fan_in: int, fan_out: int):
        limit = sqrt(2.0 / fan_in + fan_out)
        return self._generate(fan_in, fan_out, limit)

    def he_uniform(self, fan_in, fan_out):
        limit = sqrt(6.0 / fan_in)
        return self._generate(fan_in, fan_out, limit)

    def he_normal(self, fan_in, fan_out):
        limit = sqrt(2.0 / fan_in)
        return self._generate(fan_in, fan_out, limit)
