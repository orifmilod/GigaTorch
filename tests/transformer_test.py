from gigatorch.transformer import normalize
from gigatorch.tensor import Tensor
import numpy as np

import torchvision.transforms.functional as F
import torch

def test_transform():
    input = np.array([
        [ # Channel 1
            [1, 2, 3],
            [4, 5, 6], 
            [7, 8, 9]
        ]
    ], dtype=np.float32)
    mean = [0.5]
    std = [0.5]
    giga_output = normalize(Tensor(input), mean, std)
    torch_output = F.normalize(torch.Tensor(input), mean, std)

    assert (giga_output.item() == torch_output.numpy()).all()