"""Stage 18: Adam (Adaptive Moment Estimation).

Top of the optimizer chain: imports the Optimizer base / SGD / Momentum from
prior stages and adds RMSProp, Adam (momentum + 2nd-moment EMA + bias
correction), and AdamW. Optimizers consume p.grad; they never call backward().
NumPy + stdlib only.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

# Optimizer base/SGD (14), Momentum (17); Adam subclasses the base.
from dlfs import stage_import

Stage14_Optimizer, Stage14_SGD = stage_import("stage_14", "Optimizer", "SGD")
Stage17_Momentum = stage_import("stage_17", "Momentum")

# Canonical re-exports so the whole optimizer family imports from here.
Optimizer = Stage14_Optimizer
SGD = Stage14_SGD
Momentum = Stage17_Momentum


class RMSProp(Stage14_Optimizer):
    """RMSProp: per-parameter step scaled by an EMA of the squared gradient.

    g = p.grad + weight_decay*p.data; v = beta*v + (1-beta)*g**2;
    p.data -= lr * g / (sqrt(v) + eps).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        beta: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # Plumbing (provided): store hyper-params and allocate one EMA buffer per param.
        super().__init__(params)
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Apply one in-place RMSProp update to every parameter; leave p.grad untouched."""
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.v[i] = self.beta * self.v[i] + (1.0 - self.beta) * (g ** 2)
            p.data = p.data - self.lr * g / (self.v[i] ** 0.5 + self.eps)


class Adam(Stage14_Optimizer):
    """Adam: 1st-moment EMA + 2nd-moment EMA + bias correction. On step t with g:
        m = beta1*m + (1-beta1)*g;  v = beta2*v + (1-beta2)*g**2
        m_hat = m/(1-beta1**t);  v_hat = v/(1-beta2**t)
        p.data -= lr * m_hat / (sqrt(v_hat) + eps)
    Coupled weight decay (folded into g); see AdamW for decoupled."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # Plumbing (provided): store hyper-params, init step counter, allocate m/v buffers.
        super().__init__(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _effective_grad(self, p):
        return p.grad + self.weight_decay * p.data

    def step(self) -> None:
        """Apply one in-place Adam update (increment t, update m/v with bias correction, step p.data)."""
        self.t += 1
        for i, p in enumerate(self.params):
            g = self._effective_grad(p)
            # First moment (mean) estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            # Second moment (variance) estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data = p.data - self.lr * (m_hat / (v_hat ** 0.5 + self.eps))


class AdamW(Adam):
    """Adam with decoupled weight decay: decay applied to p.data directly, not folded into g.

    Same constructor signature as Adam.
    """

    def _effective_grad(self, p):
        """AdamW's adaptive update uses the raw gradient only (decay is decoupled)."""
        return p.grad

    def step(self) -> None:
        """Run the standard Adam update, then apply decoupled weight decay to p.data."""
        super(AdamW, self).step()
        for p in self.params:
            p.data -= self.lr * p.data * self.weight_decay
