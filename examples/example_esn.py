#!/usr/local/bin/python
# pylint: disable=not-callable,unsubscriptable-object
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader

from aptorch.esn import Reservoir, Ridge, ESN
from aptorch.data import narma10
from aptorch.metrics import nrmse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


if __name__ == "__main__":
    input_size = 1
    hidden_size = 5
    output_size = 1
    batch_size = 16
    n_samples = 1500

    X_design_set, y_design_set = narma10(seq_len=n_samples)

    train_size = int(n_samples*0.8)
    X_train, y_train = X_design_set[:train_size], y_design_set[:train_size]
    X_test, y_test = X_design_set[train_size:], y_design_set[train_size:]

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)
    print(f"train_set={len(train_set)}, test_set={len(test_set)}")

    reservoir = Reservoir(input_size=input_size,
                          hidden_size=hidden_size,
                          omega_in=1.0,
                          rho=0.99,
                          alpha=0.9)
    readout = Ridge(output_size=output_size, l2=1e-4)
    esn = ESN(reservoir, readout, washout=100).to(device)

    ###########################################################################
    # train
    ###########################################################################
    esn.train()
    train_loader = DataLoader(train_set, batch_size=batch_size)
    for epoch, (x, y) in tqdm(
            enumerate(train_loader), total=len(train_loader), desc="train"):
        esn(x.to(device), y.to(device))
    esn.reservoir.reset_state()

    ###########################################################################
    # eval on training set
    ###########################################################################
    esn.eval()
    train_losses = []
    train_loader = DataLoader(train_set, batch_size=batch_size)
    for x, y in train_loader:
        y_pred, _ = esn(x.to(device), y.to(device))
        loss = nrmse(y_pred.cpu(), y)
        train_losses.append(loss)
    esn.reservoir.reset_state()
    train_score = np.mean(train_losses)

    ###########################################################################
    # eval on test set
    ###########################################################################
    test_losses = []
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for x, y in test_loader:
        y_pred, _ = esn(x.to(device), y.to(device))
        loss = nrmse(y_pred.cpu(), y)
        test_losses.append(loss)
    esn.reservoir.reset_state()
    test_score = np.mean(test_losses)

    print(f"nrmse: train={train_score:.4f}, test={test_score:.4f}")
