import math
from functools import total_ordering
import numpy as np


@total_ordering
class Tensor:
    """stores a single scalar value and its gradient"""

    def __init__(self, input, _parents=[], _op=""):
        self.data = input.data if isinstance(
            input, Tensor) else np.array(input)
        self.grad = 0.0
        self._backprop = lambda: None
        self._parents = _parents
        self._op = _op

    def __repr__(self):
        return f"d:{self.data}"

    def item(self) -> np.ndarray:
        return self.data

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
        # Setting the cost node as derivative of cost to itself is 1 (dC/dC)
        self.grad = 1.0
        for node in reversed(topo):
            node._backprop()

    def append(self, *args):
        self.data = np.append(self.data, *args)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        return Tensor(data)

    def transpose(self, *axes):
        data = self.data.transpose(*axes)
        return Tensor(data)

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, [self])

        def _backprop():
            # Broadcast the output gradient back to the shape of the input
            grad = out.grad
            if np.ndim(grad) == 0:
                self.grad += np.ones_like(self.data) * grad
            else:
                # broadcast output gradient to input shape
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backprop = _backprop
        return out

    def mean(self, axis=None, keepdims=False):
        data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(data, [self])

        def _backprop():
            # Gradient of mean is 1/N broadcasted to input shape
            grad = out.grad
            # number of elements reduced along the operation
            if axis is None:
                n = self.data.size
            else:
                # compute number of elements in the reduced axes
                # using numpy to get shape after reduction
                tmp = self.data.mean(axis=axis, keepdims=keepdims)
                n = int(self.data.size / tmp.size)
            if np.ndim(grad) == 0:
                self.grad += np.ones_like(self.data) * (grad / n)
            else:
                self.grad += np.broadcast_to(grad / n, self.data.shape)

        out._backprop = _backprop
        return out

    def exp(self):
        return Tensor(np.exp(self.data))

    def log(self):
        return Tensor(np.log(self.data))

    def max(self):
        return Tensor(self.data.max())

    @staticmethod
    def zeros(*size, **kwargs):
        data = np.zeros(*size, **kwargs)
        return Tensor(data)

    @staticmethod
    def randn(*size):
        data = np.random.randn(*size)
        return Tensor(data)

    def tanh(self):
        # Use numpy's vectorized tanh to support array inputs
        out = Tensor(np.tanh(self.data), [self])

        # Backpropagation for tanh operation: derivative is 1 - tanh(x)^2
        def _backprop():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backprop = _backprop
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), [self])

        def _backprop():
            self.grad += (out.data > 0) * out.grad

        out._backprop = _backprop
        return out

    def to(self, new_type):
        return Tensor(self.data.astype(new_type))

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data + other.data, [self, other])

        # Backward propagation for addition operation
        def _backprop():
            # (Derivative with respect to itself) * output gradient
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad  # same here

        output._backprop = _backprop
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data * other.data, [self, other])

        # Backward propagation for multiplication operation
        def _backprop():
            # (Derivative with respect to itself) * output gradient
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad  # same here

        output._backprop = _backprop
        return output

    def __float__(self):
        return float(self.data)

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data**other.data, [self, other])

        def _backprop():
            # d/dself x**y = y * x**(y-1)
            self.grad += (other.data * self.data **
                          (other.data - 1)) * output.grad
            # d/dother x**y = x**y * log(x)
            # guard against non-positive x for log by using a safe substitute
            try:
                log_x = np.log(self.data)
            except Exception:
                # fallback: where self.data <= 0, use 0 (no meaningful gradient for negative/zero base in this simple impl)
                log_x = np.where(self.data > 0, np.log(self.data), 0.0)
            other.grad += (output.data * log_x) * output.grad

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

    def __len__(self):
        return self.data.shape[0]
