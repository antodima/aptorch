# pylint: disable=not-callable
from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList


class Splines(Module):
    """A layer that computes the B-Splines.
    https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 4,
        basis_order: int = 3,
        basis_range: Tuple[int, int] = (-1, 1),
        n_intervals: int = 5,
    ):
        """Splines initialization.

        Parameters
        ----------
        input_size : int, optional
            the number of input features, by default 1
        output_size : int, optional
            the number of output features, by default 4
        basis_order : int, optional
            the order of the piecewise polynomial, by default 3
        basis_range : Tuple[int, int], optional
            the range of the polynomial domain, by default (-1, 1)
        n_intervals : int, optional
            the number of pieces of the approximated polynomial, by default 5
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.basis_order = basis_order
        self.basis_range = basis_range
        self.basis_func = torch.nn.SiLU()
        self.n_intervals = n_intervals

        padding_size = 2 * basis_order + 1
        grid = torch.linspace(
            start=basis_range[0],
            end=basis_range[1],
            steps=n_intervals + padding_size,
        ).expand(input_size, -1).contiguous()
        self.register_buffer("grid", grid)

        self.basis_weights = Parameter(torch.Tensor(output_size, input_size))
        self.splines_weights = Parameter(
            torch.Tensor(output_size, input_size, n_intervals+basis_order))

    def forward(self, x: Tensor) -> Tensor:
        """Computes the basis sum of the splines.

        Parameters
        ----------
        x : Tensor
            the input tensor (batch_size, input_size)

        Returns
        -------
        Tensor
            the values of the polynomial for the input x (batch_size, output_size)
        """
        basis_out = F.linear(
            self.basis_func(x), self.basis_weights)  # (batch_size, output_size)

        # computes the basis for the input
        x = x.unsqueeze(-1)
        basis = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).float()
        for k in range(1, self.basis_order + 1):
            basis = (
                (x - self.grid[:, : -(k + 1)])
                / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)])
                * basis[:, :, :-1]
            ) + (
                (self.grid[:, k + 1:] - x)
                / (self.grid[:, k + 1:] - self.grid[:, 1:(-k)])
                * basis[:, :, 1:]
            )
        basis = basis.contiguous()  # (batch_size, input_size, n_intervals + basis_order)

        spline_out = F.linear(
            basis.view(x.size(0), -1),
            self.splines_weights.view(self.output_size, -1),
        )

        out = basis_out + spline_out  # (batch_size, output_size)
        return out


class KAN(Module):

    def __init__(
        self,
        layers: List[int],
        basis_order: int = 3,
        basis_range: Tuple[int, int] = (-1, 1),
        n_intervals: int = 5,
    ) -> None:
        super().__init__()
        self.layers = ModuleList()
        for input_size, output_size in zip(layers, layers[1:]):
            self.layers.append(
                Splines(
                    input_size=input_size,
                    output_size=output_size,
                    basis_order=basis_order,
                    basis_range=basis_range,
                    n_intervals=n_intervals,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
