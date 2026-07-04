"""Stage 17: Momentum.

SGD with a velocity buffer (heavy-ball momentum) and an optional Nesterov
variant, built by subclassing the plain ``SGD`` from stage_14.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Tensor (stage_08 engine via stage_12); SGD (stage_14).
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage14_SGD = stage_import("stage_14", "SGD")

# Re-export the autodiff Tensor under its canonical public name.
Tensor = Stage12_Tensor


class SGDMomentum(Stage14_SGD):
    """SGD with momentum (heavy-ball) and an optional Nesterov variant.

    Update with ``g = p.grad``:
        v <- beta * v + g
        p <- p - lr * v                  # heavy-ball (nesterov=False)
        p <- p - lr * (g + beta * v)     # nesterov=True
    Extends stage_14 ``SGD`` (same params/lr contract, inherited zero_grad).
    ``beta=0`` reduces exactly to plain SGD.
    """

    def __init__(
        self,
        params: Iterable["Tensor"],
        lr: float,
        beta: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        # Defer params/lr setup to the stage_14 SGD ctor (single source of truth).
        super().__init__(params, lr)
        assert beta >= 0 and beta < 1
        self.beta = beta
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Apply one momentum update to every parameter in place.

        Velocity buffers persist across calls (write updated v back into
        self.velocities). Does NOT zero grads (that's the inherited zero_grad).
        """
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.velocities[i] = self.beta * self.velocities[i] + p.grad
                if self.nesterov:
                    p.data -= self.lr * (p.grad + self.beta * self.velocities[i])
                else:
                    p.data -= self.lr * self.velocities[i]

    def reset(self) -> None:
        """Zero all velocity buffers; leaves p.data and p.grad untouched."""
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def __repr__(self) -> str:
        return f"SGDMomentum({self.lr=}, {self.beta=}, {self.nesterov=})"


# Backwards-compatible public alias used by the tests.
Momentum = SGDMomentum
