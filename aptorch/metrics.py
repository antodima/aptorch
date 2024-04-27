# pylint: disable=redefined-builtin
import torch
from torch import Tensor
import torch.nn.functional as F


def nrmse(
    input: Tensor,
    target: Tensor,
) -> float:
    """Normalized Root Mean Squared Error (RMSE) score.

    Parameters
    ----------
    input : Tensor
        The tensor of predictions
    target : Tensor
        The tensor of the targets

    Returns
    -------
    float
        The RMSE score.
    """
    rmse = torch.sqrt(F.mse_loss(input, target)).item()
    return rmse / (target.max() - target.min())
