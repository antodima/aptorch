from typing import Tuple

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
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
    x = torch.nn.init.uniform_(torch.empty(seq_len+lag), a=0.0, b=0.5).tolist()
    y = torch.zeros(seq_len+lag).tolist()
    for n in range(lag, seq_len):
        y[n] = alpha * y[n-1] + beta * y[n-1] * (
            y[n-1] + y[n-2] + y[n-3] + y[n-4] + y[n-5] +
            y[n-6] + y[n-7] + y[n-8] + y[n-9] + y[n-10]
        ) + gamma * x[n-10] * x[n-1] + delta

    inputs: Tensor = FloatTensor(x[lag:]).view(-1, 1)
    targets: Tensor = FloatTensor(y[lag:]).view(-1, 1)
    return inputs, targets


class DivinaCommediaDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["text"])

    def __getitem__(self, index):
        return self.dataset["text"][index]


def divina_commedia():
    dataset = load_dataset("maiurilorenzo/divina-commedia", split="train")
    train_size = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    return train_dataset, test_dataset


def divina_commedia_tokenizer(dataset):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_token="[PAD]", pad_id=0)
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[MASK]"])
    trainer = BpeTrainer()

    tokenizer.train_from_iterator(
        dataset["text"],
        trainer=trainer,
    )

    return tokenizer
