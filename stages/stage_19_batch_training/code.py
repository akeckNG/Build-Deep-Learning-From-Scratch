"""Stage 19: Mini-batch training.

Additive on top of stage_15's full-batch driver: split the data into mini-batches
and run forward/backward/step once per batch. Adds only the data-feeding layer
(``iterate_minibatches`` + ``train_minibatch``) plus noise/analysis helpers.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Reused framework: autodiff, model, loss, optimizers + full-batch driver.
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage11_MLP = stage_import("stage_11", "MLP")
Stage12_mse_loss = stage_import("stage_12", "mse_loss")
Stage14_Optimizer, Stage14_SGD = stage_import("stage_14", "Optimizer", "SGD")
Stage18_Adam = stage_import("stage_18", "Adam")
Stage15_accuracy, Stage15_train = stage_import("stage_15", "accuracy", "train")

# Re-export under canonical names for this stage's callers and later stages.
Tensor = Stage12_Tensor
MLP = Stage11_MLP
mse_loss = Stage12_mse_loss
Optimizer = Stage14_Optimizer
SGD = Stage14_SGD
Adam = Stage18_Adam
accuracy = Stage15_accuracy
train = Stage15_train


def iterate_minibatches(
    X,
    y,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generator yielding ``(X_b, y_b)`` mini-batches partitioning the dataset.
    X (N, n_in), y (N,) or (N, 1); ValueError if lengths differ or batch_size not in [1, N]."""
    X_arr = np.asarray(X.data if isinstance(X, Tensor) else X, dtype=np.float64)
    y_arr = np.asarray(y.data if isinstance(y, Tensor) else y, dtype=np.float64)
    n_samples = X_arr.shape[0]

    assert n_samples == y_arr.shape[0], "X and y must have the same number of samples."
    assert 1 <= batch_size <= n_samples, "batch_size must be between 1 and the number of samples."

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size

        # Check if we should drop the final incomplete batch
        if drop_last and end_idx > n_samples:
            break

        batch_indices = indices[start_idx:end_idx]

        # Slicing with an index array returns a copy, which is
        # necessary to provide contiguous/independent batches.
        yield X_arr[batch_indices], y_arr[batch_indices]

def train_minibatch(
    model: "Stage11_MLP",
    X,
    y,
    *,
    lr: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    optimizer: Optional["Stage14_Optimizer"] = None,
    drop_last: bool = False,
) -> Dict[str, object]:
    """Train ``model`` with mini-batch gradient descent; return loss history
    ``{"batch_loss", "epoch_loss", "steps"}`` (epoch_loss = size-weighted mean)."""
    if optimizer is None:
        optimizer = SGD(model.parameters(), lr=lr)

    batch_loss = []
    epoch_loss = []
    steps = 0

    for _ in range(epochs):
        losses = []
        sizes = []
        for X_b, y_b in iterate_minibatches(
            X, y, batch_size=batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last
        ):
            pred = model(Tensor(X_b))
            loss = mse_loss(pred, y_b.reshape(-1, 1))
            batch_loss.append(float(loss.data))
            losses.append(float(loss.data))
            sizes.append(X_b.shape[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

        # Size-weighted mean so a ragged last batch doesn't skew the epoch loss.
        epoch_loss.append(float(np.average(losses, weights=sizes)))

    return {"batch_loss": batch_loss, "epoch_loss": epoch_loss, "steps": steps}


def gradient_noise(
    model: "Stage11_MLP",
    X,
    y,
    batch_size: int,
    *,
    n_batches: int,
    seed: Optional[int] = None,
) -> float:
    """Estimate the variance of the mini-batch gradient at a fixed model: mean over
    coordinates of the per-coordinate variance (should fall like sigma**2/batch_size)."""
    params = model.parameters()
    grads = []
    epoch = 0

    # Cycle fresh (re-seeded) epochs until n_batches gradients are collected.
    while len(grads) < n_batches:
        epoch_seed = None if seed is None else seed + epoch
        for X_b, y_b in iterate_minibatches(
            X, y, batch_size=batch_size, shuffle=True, seed=epoch_seed
        ):
            if len(grads) >= n_batches:
                break
            for p in params:
                p.grad = np.zeros_like(p.data)
            loss = mse_loss(model(Tensor(X_b)), y_b.reshape(-1, 1))
            loss.backward()
            grads.append(np.concatenate([np.asarray(p.grad).ravel() for p in params]))
        epoch += 1

    # Leave the model untouched: no optimizer step ran; clear the scratch grads.
    for p in params:
        p.grad = np.zeros_like(p.data)

    return float(np.stack(grads).var(axis=0).mean())


def epochs_to_threshold(history: Sequence[float], threshold: float) -> int:
    """Return the 1-based epoch index where loss first reaches ``threshold``, else -1."""
    for i, loss in enumerate(history):
        if loss <= threshold:
            return i + 1  # 1-based index
    return -1

def plot_batch_comparison(
    histories: Mapping[str, Sequence[float]],
    path: Optional[str] = None,
):
    """Plot several labelled epoch-loss curves on shared axes; save to ``path`` or return fig."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, history in histories.items():
        ax.plot(range(1, len(history) + 1), history, label=label, marker='o', markersize=3)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if path:
        plt.savefig(path)
        plt.close(fig)
    else:
        return fig