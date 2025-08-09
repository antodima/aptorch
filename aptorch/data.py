from typing import Tuple

import requests
import torch
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset


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
    lag = 10
    x = torch.nn.init.uniform_(torch.empty(seq_len + lag), a=0.0, b=0.5).tolist()
    y = torch.zeros(seq_len + lag).tolist()
    for n in range(lag, seq_len):
        y[n] = (
            alpha * y[n - 1]
            + beta
            * y[n - 1]
            * (
                y[n - 1]
                + y[n - 2]
                + y[n - 3]
                + y[n - 4]
                + y[n - 5]
                + y[n - 6]
                + y[n - 7]
                + y[n - 8]
                + y[n - 9]
                + y[n - 10]
            )
            + gamma * x[n - 10] * x[n - 1]
            + delta
        )

    inputs: Tensor = FloatTensor(x[lag:]).view(-1, 1)
    targets: Tensor = FloatTensor(y[lag:]).view(-1, 1)
    return inputs, targets


###############################################################################


def get_shakespeare_text():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url, stream=True)
    text = response.content.decode("utf-8")

    return text


class ShakespeareDataset(Dataset):
    def __init__(
        self,
        text: str,
        chat_to_token: dict,
        unk_token: str,
        input_len: int = 256,
        output_len: int = 1,
    ):
        super(ShakespeareDataset).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        self.c2i = chat_to_token
        characters = list(text)
        tokens = [self.c2i.get(c, unk_token) for c in characters]
        self.data = torch.tensor(
            [tokens[i : i + self.seq_len] for i in range(len(tokens) - self.seq_len)],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index, : self.input_len], self.data[index, self.input_len :]
        assert x.shape[-1] == self.input_len
        assert y.shape[-1] == self.output_len
        return x, y.squeeze()
