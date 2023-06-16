import math
from functools import total_ordering
import numpy as np
from torch import index_add


@total_ordering
class Tensor:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _parents=[], _op=""):
        self.data = data.data if isinstance(data, Tensor) else np.array(data)
        self.grad = 0.0
        self._backprop = lambda: None
        self._parents = _parents
        self._op = _op

    def __repr__(self):
        return f"d:{self.data}"

    def relu(self):
        out = Tensor(max(self.data, 0), [self], "ReLU")

        def _backprop():
            self.grad += (out.data > 0) * out.grad

        out._backprop = _backprop

        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data + other.data, [self, other], "+")

        # Backward propagation for addition operation
        def _backprop():
            self.grad += (
                1.0 * output.grad
            )  # (Derivative with respect to itself) * output gradient
            other.grad += 1.0 * output.grad  # same here

        output._backprop = _backprop
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data * other.data, [self, other], "*")

        # Backward propagation for multiplication operation
        def _backprop():
            self.grad += (
                other.data * output.grad
            )  # (Derivative with respect to itself) * output gradient
            other.grad += self.data * output.grad  # same here

        output._backprop = _backprop
        return output

    def __float__(self):
        return float(self.data)

    def to(self, new_type):
        return Tensor(self.data.astype(new_type))

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data**other.data, [self], f"**{other.data}")

        def _backprop():
            self.grad += (
                other.data * self.data ** (other.data - 1)
            ) * output.grad  # (derivative of the power) * (output gradient)

        output._backprop = _backprop
        return output

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self.data == (other.data if isinstance(other, Tensor) else other)

    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, Tensor) else other)

    def __getitem__(self, indices):
        return Tensor(self.data[indices])

    def __setitem__(self, indices, value):
        self.data[indices] = value.data if isinstance(value, Tensor) else value

    def item(self):
        return self.data

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        return Tensor(data)

    def transpose(self, *axes):
        data = self.data.transpose(*axes)
        return Tensor(data)

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        return Tensor(data)

    def mean(self, axis=None, keepdims=False):
        data = self.data.mean(axis=axis, keepdims=keepdims)
        return Tensor(data)

    @staticmethod
    def zeros(*size, **kwargs):
        data = np.zeros(*size, **kwargs)
        return Tensor(data)

    @staticmethod
    def randn(*size):
        data = np.random.randn(*size)
        return Tensor(data)

    def backprop(self):
        # Used for calculating gradient of the nodes in order
        def _build_topological_sort(node, topo=[], visited=set()):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    _build_topological_sort(parent, topo, visited)

                topo.append(node)

            return topo

        topo = _build_topological_sort(self)

        # Propagate the gradient backprops
        self.grad = (
            1.0  # Setting the cost node as derivative of cost to itself is 1 (dC/dC)
        )
        for node in reversed(topo):
            node._backprop()

    def append(self, *args):
        self.data = np.append(self.data, *args)

    def tanh(self):
        # other = other if isinstance(other, Tensor) else Tensor(other)
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Tensor(t, [self], "tanh")

        # Backpropagation for tanh operation
        def _backprop():
            self.grad += (
                1 - t**2
            ) * output.grad  # (derivative of tanh) * output gradient https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives

        output._backprop = _backprop
        return output
