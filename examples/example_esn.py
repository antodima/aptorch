#!/usr/local/bin/python
# pylint: disable=not-callable,unsubscriptable-object
from typing import Dict, Generator
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

from aptorch.esn import Reservoir, Ridge, ESN
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


def build_vanilla_esn(**config) -> Module:
    reservoir = Reservoir(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        omega_in=config["omega_in"],
        rho=config["rho"],
        alpha=config["alpha"],
    )
    readout = Ridge(
        output_size=config["output_size"],
        l2=config["l2"],
    )
    model = ESN(reservoir, readout, washout=config["washout"])
    return model


def train_vanilla_esn(model: ESN, data_set: TensorDataset, **config) -> ESN:
    model.train()
    train_loader = DataLoader(data_set, batch_size=config["batch_size"])
    for x, y in tqdm(train_loader, total=len(train_loader), desc="train"):
        model(x.to(device), y.to(device))
    model.reservoir.reset_state()
    return model


def evaluate_vanilla_esn(
    model: ESN,
    train_set: TensorDataset,
    test_set: TensorDataset,
    **config,
):
    model.eval()
    train_losses = []
    train_loader = DataLoader(train_set, batch_size=config["batch_size"])
    for x, y in train_loader:
        y_pred, _ = model(x.to(device), y.to(device))
        loss = nrmse(y_pred.cpu(), y)
        train_losses.append(loss)
    model.reservoir.reset_state()
    train_score = np.mean(train_losses)

    test_losses = []
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])
    for x, y in test_loader:
        y_pred, _ = model(x.to(device), y.to(device))
        loss = nrmse(y_pred.cpu(), y)
        test_losses.append(loss)
    model.reservoir.reset_state()
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
        "input_size": [1],
        "output_size": [1],
        "hidden_size": [8, 16, 32, 64, 128, 256],
        "omega_in": [1.0, 0.9],
        "rho": [0.99],
        "alpha": [1.0, 0.8, 0.5, 0.3],
        "l2": [None, 5e-3, 5e-4],
        "washout": [0, 10, 100],
        "batch_size": [8, 16, 32, 64],
    }
    for config in grid_search(**param_space):
        esn = build_vanilla_esn(**config)
        esn = esn.to(device)

        esn = train_vanilla_esn(esn, train_set, **config)
        scores = evaluate_vanilla_esn(esn, train_set, test_set, **config)
        print(f"config={config}, scores={scores}")
