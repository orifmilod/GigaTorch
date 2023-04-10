import math

class Value:
  """ stores a single scalar value and its gradient """

  def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0.0
    self._backprop = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)

    output = Value(self.data + other.data, (self, other), '+')

    # Backward propagation for addition operation
    def _backprop():
      self.grad += 1.0 * output.grad  # (Derivative with respect to itself) * output gradient
      other.grad += 1.0 * output.grad # same here

    output._backprop = _backprop
    return output

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)

    output = Value(self.data * other.data, (self, other), '*')

    # Backward propagation for multiplication operation
    def _backprop():
      self.grad += other.data * output.grad  # (Derivative with respect to itself) * output gradient
      other.grad += self.data * output.grad  # same here

    output._backprop = _backprop

    return output

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    output = Value(self.data ** other, (self, other), f'**{other}')

    def _backprop():
        self.grad += (other * self.data ** (other - 1)) * output.grad # (derivative of the power) * (output gradient)

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
    return self * other ** -1

  def __rtruediv__(self, other):
    return other * self ** -1

  def backprop(self):
    # Used for calculating gradient of the nodes in order
    def _build_topological_sort(node, topo = [], visited = set()):
      if node not in visited:
        visited.add(node)
        for child in node._prev:
          _build_topological_sort(child, topo, visited)

        topo.append(node)

      return topo

    topo = _build_topological_sort(self)

    # Propagate the gradient backprops
    self.grad = 1.0 # Setting the cost node as derivative of cost to itself is 1 (dC/dC)
    for node in reversed(topo):
      node._backprop()


  def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    output = Value(t, (self, ), 'tanh')

    #Backpropagation for tanh operation
    def _backprop():
        self.grad = (1 - t ** 2) * output.grad # (derivative of tanh) * output gradient https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives

    self._backprop = _backprop
    return output

    def squared(self):
      # implement squared loss computation
      pass
