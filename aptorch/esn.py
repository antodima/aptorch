from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

from aptorch.init import normal, uniform


class Reservoir(Module):
    """Reservoir Layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        omega_in: float = 1.0,
        rho: float = 0.99,
        alpha: float = 1.0,
        input_init_fn: Callable = uniform(),
        hidden_init_fn: Callable = normal(),
    ):
        super().__init__()
        assert alpha <= 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.omega_in = omega_in
        self.rho = rho
        self.alpha = alpha

        # model parameters
        self.x = Parameter(
            torch.zeros(self.hidden_size),
            requires_grad=False,
        )
        self.weight_ih = Parameter(
            input_init_fn(shape=(self.hidden_size, self.input_size)),
            requires_grad=False,
        )
        self.bias_ih = Parameter(
            input_init_fn(shape=(self.hidden_size)),
            requires_grad=False,
        )
        self.weight_hh = Parameter(
            hidden_init_fn(shape=(self.hidden_size, self.hidden_size)),
            requires_grad=False,
        )

        # scale parameters matrices
        self.weight_ih.data = self._rescale(
            self.weight_ih.data, scale=self.omega_in)
        self.bias_ih.data = self._rescale(
            self.bias_ih.data, scale=self.omega_in)
        self.weight_hh.data = self._rescale(
            self.weight_hh.data, spectral_radius=self.rho)

    @staticmethod
    def _rescale(
        W: Tensor,
        spectral_radius: Optional[float] = None,
        scale: Optional[float] = None,
    ) -> Tensor:
        if spectral_radius is not None:
            W.div_(torch.linalg.eigvals(W).abs().max()).mul_(
                spectral_radius).float()

        if scale is not None:
            W.mul_(scale).float()

        return W

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
    ) -> Optional[Tensor]:
        """Performs feedforward pass.

        Parameters
        ----------
        input : Tensor
            The input data (seq_len, batch_size, input_size)

        Returns
        -------
        Tuple[Optional[Tensor], Tensor]
            The aggregated states (hidden_size, seq_len) and the last state.
        """
        x = self.x.data
        x = x.to(self.x)

        states: Optional[Tensor] = None
        for u in input:
            net_ih = F.linear(u.to(self.weight_ih),
                              self.weight_ih, self.bias_ih)
            net_hh = F.linear(x, self.weight_hh)
            x_t = torch.tanh(net_ih + net_hh)
            x = (1 - self.alpha) * x + self.alpha * x_t
            states = x.view(-1, 1) \
                if states is None \
                else torch.cat((states, x.view(-1, 1)), dim=-1)

        self.x.data = x
        return states

    def reset_state(self):
        self.x.data = torch.zeros(self.hidden_size).to(self.x)


class Ridge(Module):
    """Ridge Regression Layer.
    """

    def __init__(
        self,
        output_size: int = 1,
        l2: Optional[float] = None,
    ):
        super().__init__()
        self.output_size = output_size
        self.A = Parameter(requires_grad=False)  # (seq_len, hidden_size)
        self.B = Parameter(requires_grad=False)  # (hidden_size, hidden_size)
        self.W = Parameter(requires_grad=False)  # (output_size, hidden_size)
        self.l2 = l2

    def forward(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """Performs feedforward pass.

        Parameters
        ----------
        X : Tensor
            The input tensor (seq_len, hidden_size)
        y : Optional[Tensor]
            The target tensor (seq_len, output_size), default to None
        """
        if self.training:
            batch_A, batch_B = (y.T @ X), (X.T @ X)
            self.A.data, self.B.data = (self.A.data + batch_A, self.B.data +
                                        batch_B) if len(self.A) is None else (batch_A, batch_B)

            A, B = self.A.data, self.B.data + \
                torch.eye(self.B.data.shape[0]).to(self.B) * \
                self.l2 if self.l2 else self.B.data

            self.W.data = A @ B.pinverse()  # (output_size, hidden_size)

        y_pred = F.linear(X, self.W.data)  # (seq_len, output_size)
        return y_pred


class ESN(Module):
    """Vanilla Echo State Network.
    """

    def __init__(
        self,
        reservoir: Reservoir,
        readout: Module,
        washout: int = 0,
    ):
        super().__init__()
        self.reservoir = reservoir
        self.readout = readout
        self.washout = washout

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tuple[Optional[Tensor], Tensor]:
        input = torch.atleast_2d(input)
        target = torch.atleast_2d(target)

        # computes hidden states
        states = self.reservoir(input)

        # apply washout
        n_states = states.shape[1]
        washed_states = states[:, self.washout:]
        washed_target = target[self.washout:, :]
        self.washout = max(0, (self.washout - n_states))

        output = None
        if washed_states.shape[1] > 0:
            output = self.readout(washed_states.T, washed_target)

        return output, washed_target
