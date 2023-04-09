import math

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data},{self.grad})"

  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    # Backward propagation for addition operation
    def _backward():
      self.grad += 1.0 * out.grad  # Derivative with respect to itself * output gradient
      other.grad += 1.0 * out.grad # same goes with the other variable

    out._backward = _backward
    return out

  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    # Backward propagation for multiplication operation
    def _backward():
      self.grad += other.grad * out.grad  # Derivative with respect to itself * output gradient
      other.grad += self.grad * out.grad # same goes with the other variable

    out._backward = _backward

    return out

  def backward(self):
    # Topological sort, used for backpropagation 

    def _build_topo(node, topo = [], visited = set()):
      if node not in visited:
        visited.add(node)
        for child in node._prev:
          _build_topo(child, topo, visited)

        topo.append(node)

      return topo

    topo = _build_topo(self)

    # Propagate the gradient backwards

    self.grad = 1.0 # Setting the Cost node as 1
    for node in reversed(topo):
      node._backward()


  def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self, ), 'tanh')

    #Backpropagation for tanh operation
    def _backward():
        self.grad = (1 - t ** 2) * out.grad # (derivative of tanh) * output gradient https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives

    self._backward = _backward
    return out

    def squared(self):
      # implement squared loss computation
      pass


