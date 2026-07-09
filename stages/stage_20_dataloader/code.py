"""Stage 20: DataLoader.

Packages stage_19's one-shot batch logic as reusable objects: an indexable
``Dataset``, a re-iterable ``DataLoader`` (iterator protocol, re-shuffles each
epoch), ``train_val_split``, and a ``train_with_loader`` driver. No new gradients.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import math
from dlfs import stage_import

# Tensor (stage_08 engine via stage_12); MLP/mse_loss/SGD/train_minibatch re-exported by stage_19.
Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage19_MLP, Stage19_mse_loss, Stage19_SGD, Stage19_train_minibatch = stage_import(
    "stage_19", "MLP", "mse_loss", "SGD", "train_minibatch"
)

# Re-export imported pieces under canonical public names.
Tensor = Stage12_Tensor
MLP = Stage19_MLP
mse_loss = Stage19_mse_loss
SGD = Stage19_SGD
train_minibatch = Stage19_train_minibatch


class Dataset:
    """Map-style dataset wrapping (X, y) via __len__ + __getitem__. X (N, n_in),
    y (N,) or (N, 1); ValueError if lengths differ."""

    def __init__(self, X, y) -> None:
        # TODO: store X, y as float64 ndarrays; validate matching first dim.
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float64)
        self.y = y.astype(np.float64)


    def __len__(self) -> int:
        """Return N, the number of examples."""
        return self.X.shape[0]

    def __getitem__(self, index):
        """Return ``(X[index], y[index])``; index may be int, slice, or int array."""
        return (self.X[index], self.y[index])


class DataLoader:
    """Iterate a ``Dataset`` in mini-batches, re-shuffling each epoch; each
    ``__iter__`` is a fresh pass. batch_size must be >= 1 (else ValueError)."""

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        assert batch_size >= 1
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
    def __len__(self) -> int:
        """Return batches per epoch: N // batch_size (drop_last) else ceil(N / batch_size)."""
        N = len(self.dataset)
        return N // self.batch_size if self.drop_last else math.ceil(N / self.batch_size)

    def __iter__(self) -> Iterator[Tuple["Stage12_Tensor", "Stage12_Tensor"]]:
        """Yield ``(X_b, y_b)`` Tensor batches for one epoch (fresh order, optional
        drop_last). X_b: (b, n_in), y_b: (b,) or (b, 1)."""
        dataset_len = len(self.dataset)
        indices = np.arange(dataset_len) 

        if self.shuffle:
            np.random.default_rng(seed=self.seed).shuffle(indices)

        indices_batch = np.split(indices, np.arange(self.batch_size, len(indices), self.batch_size))

        if self.drop_last and len(indices_batch[-1]) < self.batch_size:
            indices_batch = indices_batch[:-1]

        for batch in indices_batch:
            yield Tensor(self.dataset[batch][0]), Tensor(self.dataset[batch][1])


def train_val_split(
    X,
    y,
    val_frac: float,
    *,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(X, y)`` into disjoint train/val (last round(val_frac*N) held out).
    Returns (X_tr, y_tr, X_val, y_val); requires 0 <= val_frac < 1."""
    assert 0 <= val_frac < 1
    N = X.shape[0]
    n_val = int(round(val_frac * N))

    indices = np.random.default_rng(seed=seed).permutation(N)
    train_idx, val_idx = indices[: N - n_val], indices[N - n_val :]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_with_loader(
    model: "Stage19_MLP",
    loader: "DataLoader",
    *,
    lr: float = 0.1,
    epochs: int = 50,
    optimizer: Optional["Stage19_SGD"] = None,
) -> Dict[str, object]:
    """Train ``model`` by iterating ``loader`` for ``epochs`` epochs (forward ->
    mse_loss -> backward -> step -> zero_grad). Returns
    {"batch_loss": list, "epoch_loss": list, "steps": int}."""
    # TODO: build optimizer if None, then run the per-epoch / per-batch loop.
    if optimizer is None:
        optimizer = SGD(model.parameters(), lr=lr)

    steps = 0
    epoch_loss = []
    batch_loss = []

    for _ in range(epochs):
        current_epoch_loss = []
        for (X_b, y_b) in loader:
            pred = model(X_b)
            loss = mse_loss(pred, y_b)
            batch_loss.append(float(loss.data))
            current_epoch_loss.append(float(loss.data))
            loss.grad = 1.0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

        epoch_loss.append(np.mean(current_epoch_loss).item())
    
    return {"epoch_loss": epoch_loss, "batch_loss": batch_loss, "steps": steps}

