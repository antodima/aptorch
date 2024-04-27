from typing import Tuple
import torch
from torch import FloatTensor, Tensor


def narma10(
    seq_len: int,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 1.5,
    delta: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """https://arxiv.org/pdf/1401.2224.pdf
    https://aureservoir.sourceforge.net/narma10_8py-example.html
    """
    n = 10
    x = torch.nn.init.uniform_(torch.empty(seq_len), a=0.0, b=0.5).tolist()
    y = torch.zeros(seq_len).tolist()
    for n in range(10, seq_len):
        y[n] = alpha * y[n-1] + beta * y[n-1] * (
            y[n-1] + y[n-2] + y[n-3] + y[n-4] + y[n-5] +
            y[n-6] + y[n-7] + y[n-8] + y[n-9] + y[n-10]
        ) + gamma * x[n-10] * x[n-1] + delta

    inputs: Tensor = FloatTensor(x).view(-1, 1)
    targets: Tensor = FloatTensor(y).view(-1, 1)
    return inputs, targets
