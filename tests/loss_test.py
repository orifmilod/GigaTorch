from gigatorch.loss import cross_entropy_loss
from gigatorch.tensor import Tensor
import torch


def test_cross_entropy():
    logits = [3.2, 1.3,0.2, 0.8]
    ys = [0.775, 0.116, 0.039, 0.070] # softmax-ed
    y_pred = [1, 0, 0, 0]

    loss = cross_entropy_loss(Tensor(ys), Tensor(y_pred))
    expected = torch.nn.functional.cross_entropy(torch.Tensor(logits), torch.Tensor(y_pred))

    tol = 1e-3 # idk why the error is so high
    assert abs(loss.item() - expected.item()) < tol
