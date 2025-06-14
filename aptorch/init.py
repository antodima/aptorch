from typing import Callable

import torch
from torch.nn.init import (
    normal_,
    ones_,
    orthogonal_,
    sparse_,
    trunc_normal_,
    uniform_,
    zeros_,
)


def uniform(
    lower: float = 0.0,
    upper: float = 1.0,
) -> Callable:
    """https://pytorch.org/docs/2.2/nn.init.html#torch.nn.init.uniform_
    """
    return lambda shape: uniform_(
        torch.empty(shape),
        a=lower, b=upper
    )


def normal(
    mean: float = 0.0,
    std: float = 1.0,
) -> Callable:
    """https://pytorch.org/docs/2.2/nn.init.html#torch.nn.init.normal_
    """
    return lambda shape: normal_(
        torch.empty(shape),
        mean=mean, std=std,
    )


def trunc_normal(
    mean: float = 0.0,
    std: float = 1.0,
    min_cutoff: float = -2.0,
    max_cutoff: float = 2.0,
) -> Callable:
    """https://pytorch.org/docs/2.2/nn.init.html#torch.nn.init.trunc_normal_
    """
    return lambda shape: trunc_normal_(
        torch.empty(shape),
        mean=mean, std=std,
        a=min_cutoff, b=max_cutoff,
    )


def orthogonal(
    gain: float = 1.0,
) -> Callable:
    """https://pytorch.org/docs/2.2/nn.init.html#torch.nn.init.orthogonal_
    """
    return lambda shape: orthogonal_(
        torch.empty(shape),
        gain=gain,
    )


def ring(
    gain: float = 1.0,
) -> Callable:
    """https://doi.org/10.1109/IJCNN48605.2020.9206723
    """
    return lambda shape: torch.eye(shape[0]).roll(1, 0) * gain


def antisymmetric(
    gain: float = 1.0,
) -> Callable:
    """https://arxiv.org/abs/2203.09382
    """
    def antisymmetric_(dim: int, gain: float):
        w = uniform_(torch.empty(dim, dim))
        return (w - w.T) * gain

    return lambda shape: antisymmetric_(dim=shape[0], gain=gain)


def sparse(
    sparsity: float = 0.1,
    std: float = 0.01,
) -> Callable:
    """https://pytorch.org/docs/2.2/nn.init.html#torch.nn.init.sparse_
    """
    return lambda shape: sparse_(
        torch.empty(shape),
        sparsity=sparsity, std=std,
    )


def ones() -> Callable:
    """https://pytorch.org/docs/2.2/nn.init.html#torch.nn.init.ones_
    """
    return lambda shape: ones_(torch.empty(shape))


def zeros() -> Callable:
    """https://pytorch.org/docs/2.2/nn.init.html#torch.nn.init.zeros_
    """
    return lambda shape: zeros_(torch.empty(shape))
