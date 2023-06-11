from mytorch.tensor import Tensor

import torch


def test_addition():
    a = Tensor(3.0)
    b = a + a

    b.grad = 1.0
    b.backprop()

    assert b.grad == 1.0
    assert b.data == 6.0

    assert a.grad == 2.0
    assert a.data == 3.0


def test_multiplication():
    a = Tensor(2.0)
    b = Tensor(4.0)
    c = a * b

    c.grad = 1.0
    c.backprop()

    assert c.data == 8.0
    assert c.grad == 1.0

    assert a.grad == 4.0
    assert b.grad == 2.0


def test_complex():
    a = Tensor(-2.0)
    b = Tensor(3.0)

    c = a + b  # 1
    d = a * b  # -6

    e = c * d

    e.grad = 1.0
    e.backprop()

    assert e.data == -6.0
    assert e.grad == 1.0

    assert c.data == 1.0
    assert c.grad == -6.0

    assert d.data == -6.0
    assert d.grad == 1.0

    assert a.data == -2.0
    assert a.grad == -3.0

    assert b.data == 3.0
    assert b.grad == -8.0


def test_sanity_check():
    x = Tensor(-4.0)
    x.label = "x"
    z = 2 * x + 2 + x
    z.label = "z"
    q = z.relu() + z * x
    q.label = "q"
    h = (z * z).relu()
    h.label = "h"
    y = h + q + q * x
    y.label = "y"
    y.backprop()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()


def test_more_ops():
    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_sub_operation():
    a = Tensor(1.2)
    a -= 0.2
    assert a.data == 1.0

    a -= Tensor(0.5)
    assert a.data == 0.5
