from numpy import allclose
from gigatorch.loss import cross_entropy_loss, softmax, squared_loss
from gigatorch.tensor import Tensor
import torch


def test_cross_entropy():
    logits = [3.2, 1.3, 0.2, 0.8]
    ys = [0.775, 0.116, 0.039, 0.070]  # softmax-ed
    y_pred = [1, 0, 0, 0]

    loss = cross_entropy_loss(Tensor(ys), Tensor(y_pred))
    expected = torch.nn.functional.cross_entropy(
        torch.Tensor(logits), torch.Tensor(y_pred)
    )

    tol = 1e-3  # idk why the error is so high
    assert abs(loss.item() - expected.item()) < tol


def test_softmax():
    logits = Tensor([3.2, 1.3, 0.2, 0.8])
    expected = Tensor([0.77514955, 0.11593805, 0.03859242, 0.07031998])
    output = softmax(logits)

    tol = 1e-6
    assert allclose(expected.item(), output.item(), atol=tol)


def test_softmax_extreme_values():
    logits = Tensor([-123, 3.2, 1.3, 4000])
    expected = Tensor([0.0, 0.0, 0.0, 1])  # last number should be 1 and not 'nan'
    output = softmax(logits)
    assert all(expected.item() == output.item())


def test_squared_loss():
    y_pred = Tensor([0.25, 0.75, 0, 0, 0])
    ys = Tensor([1, 0, 0, 0, 0])

    loss = squared_loss(ys, y_pred)
    assert loss.item() == 1.125