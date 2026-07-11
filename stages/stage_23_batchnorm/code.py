"""Stage 23: Batch Normalization (1-D).

Self-contained module with hand-written forward/backward on raw NumPy arrays (not
the Tensor engine). Per feature: x_hat=(x-mu)/sqrt(var+eps); y=gamma*x_hat+beta.
Train uses batch stats + updates running (EMA) buffers; eval uses the buffers.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Tensor (stage_08 engine via stage_12), imported for shape/parity use.
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")


class BatchNorm1d:
    """BatchNorm over (B, num_features): standardize each column, then affine y = gamma*x_hat + beta."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        # TODO: init params, buffers, grads, cache.
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))
        self.training = True
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))
        # (x_hat, istd) from the last train forward; backward reads these.
        self.cache = None

    def train(self, mode: bool = True) -> "BatchNorm1d":
        """Set ``self.training = mode`` (True -> batch stats, update buffers);
        return self."""
        self.training = mode
        return self

    def eval(self) -> "BatchNorm1d":
        """Switch to eval mode (running buffers, no updates); return self."""
        return self.train(False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize x (B, num_features) + affine. Train: BIASED batch var, EMA-update
        buffers with UNBIASED var, cache for backward. Eval: running buffers only."""
        return self(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for :meth:`forward`."""
        assert x.shape[-1] == self.num_features

        if self.training:
            mean = np.mean(x, axis=0)         # compute the mean across all examples in the batch per each feature
            var = np.var(x, axis=0, ddof=0)   # same for var 

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1.0 - self.momentum) * self.running_var  + self.momentum * np.var(x, axis=0, ddof=1)
        else:
            mean = self.running_mean
            var  = self.running_var

        istd = 1.0 / ((var + self.eps) ** 0.5)
        x_hat = (x - mean) * istd
        if self.training:
            self.cache = (x_hat, istd)
        return self.gamma * x_hat + self.beta

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backprop dL/dy (B, num_features) -> dL/dx; set gamma_grad/beta_grad (only
        valid after a train forward). Collapsed standardization backward, with
        g_hat = grad_out * gamma:
            dx = (istd / B) * (B*g_hat - g_hat.sum(0) - x_hat*(g_hat*x_hat).sum(0))
        """
        if self.cache is None:
            raise RuntimeError("backward requires a train-mode forward first")
        x_hat, istd = self.cache
        B = grad_out.shape[0]

        # y = gamma * x_hat + beta: each param touches every row of its column,
        # so its grad is the column-sum over the batch.
        self.beta_grad = grad_out.sum(axis=0)
        self.gamma_grad = (grad_out * x_hat).sum(axis=0)

        g_hat = grad_out * self.gamma
        return (istd / B) * (
            B * g_hat - g_hat.sum(axis=0) - x_hat * (g_hat * x_hat).sum(axis=0)
        )

    def parameters(self) -> List[np.ndarray]:
        """Return learnable params [gamma, beta] (buffers are excluded)."""
        return [self.gamma, self.beta]

    def zero_grad(self) -> None:
        """Reset gamma_grad and beta_grad to zeros."""
        self.beta_grad = 0.0
        self.gamma_grad = 0.0

    def __repr__(self) -> str:
        num_features = self.num_features
        eps = self.eps
        momentum = self.momentum
        training = self.training
        return f"BatchNorm1d({num_features=}, {eps=}, {momentum=}, {training=})"
