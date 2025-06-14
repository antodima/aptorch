#!/usr/local/bin/python
# pylint: disable=not-callable,unsubscriptable-object,no-value-for-parameter
from typing import Dict, Generator
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import MSELoss
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, DataLoader

from aptorch.kan import KAN
from aptorch.data import narma10
from aptorch.metrics import nrmse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def grid_search(**kwargs) -> Generator:
    from itertools import product
    values = list(kwargs.values())
    names = list(kwargs.keys())
    configs = [dict(zip(names, elem)) for elem in product(*values)]
    for config in configs:
        yield config


def build_kan(**config) -> KAN:
    model = KAN(
        layers=config["layers"],
        basis_order=config["basis_order"],
        basis_range=config["basis_range"],
        n_intervals=config["n_intervals"],
    )
    return model


def train_kan(model: KAN, data_set: TensorDataset, **config) -> KAN:
    model.train()
    optim = RMSprop(model.parameters(), lr=config["lr"])
    loss_fn = MSELoss()
    train_loader = DataLoader(data_set, batch_size=config["batch_size"])

    for x, y in (pbar := tqdm(train_loader, total=len(train_loader), desc="train")):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y).cpu()
        loss.backward()
        optim.step(closure=None)
        pbar.set_description(f"Loss: {loss.item():.4f}")

    return model


def evaluate_kan(
    model: KAN,
    train_set: TensorDataset,
    test_set: TensorDataset,
    **config,
):
    model.eval()
    train_losses = []
    train_loader = DataLoader(train_set, batch_size=config["batch_size"])
    for x, y in train_loader:
        x = x.to(device)
        y_pred = model(x).cpu()
        loss = nrmse(y_pred, y)
        train_losses.append(loss)
    train_score = np.mean(train_losses)

    test_losses = []
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])
    for x, y in test_loader:
        x = x.to(device)
        y_pred = model(x).cpu()
        loss = nrmse(y_pred, y)
        test_losses.append(loss)
    test_score = np.mean(test_losses)

    scores = {"nrmse": (train_score, test_score)}
    return scores


if __name__ == "__main__":
    n_samples = 1500
    X_design_set, y_design_set = narma10(seq_len=n_samples)

    train_size = int(n_samples*0.8)
    X_train, y_train = X_design_set[:train_size], y_design_set[:train_size]
    X_test, y_test = X_design_set[train_size:], y_design_set[train_size:]

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)
    print(f"train_set={len(train_set)}, test_set={len(test_set)}")

    param_space = {
        "lr": [0.5, 0.05, 0.005],
        "batch_size": [8, 16, 32, 64],
        "layers": [
            [1, 4, 1]
        ],
        "basis_order": [1, 2, 3],
        "basis_range": [(-1, 1)],
        "n_intervals": [5, 10, 15],
    }
    for config in grid_search(**param_space):
        kan = build_kan(**config)
        kan = kan.to(device)

        kan = train_kan(kan, train_set, **config)
        scores = evaluate_kan(kan, train_set, test_set, **config)
        print(f"config={config}, scores={scores}")
