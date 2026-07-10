"""Stage 21: L1 & L2 regularization.

Adds a weight penalty to the loss: L_tilde = L + lam * R, where
R2 = 0.5 * sum(theta**2) (grad theta) and R1 = sum(|theta|) (grad sign(theta)).
Reuses the stage_12 Tensor engine and stage_12 losses; penalties backprop
through Tensor.backward(). Allowed: numpy (forward only), stdlib, imported Tensor/losses.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

# Tensor (12); mse/mae losses (12); cross_entropy (13, batched).
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage12_mse_loss, Stage12_mae_loss = stage_import("stage_12", "mse_loss", "mae_loss")
# cross_entropy is used on batched (B, C) logits here, so take the batched version.
Stage13_cross_entropy_loss = stage_import("stage_13", "cross_entropy_loss")

# Re-export engine + losses under canonical names for downstream stages/tests.
# Prefer stage_12's reduction-capable Tensor (sum/mean on stage_11's broadcasting backward).
Tensor = Stage12_Tensor
mse_loss = Stage12_mse_loss
mae_loss = Stage12_mae_loss
cross_entropy_loss = Stage13_cross_entropy_loss


def _abs(t: "Stage12_Tensor") -> "Stage12_Tensor":
    """Elementwise |t| via the relu identity |t| = relu(t) + relu(-t), so the
    sub-gradient sign(t) flows through autodiff (do NOT use np.abs on the graph)."""
    return t.relu() + (-t).relu()


def l2_penalty(params: Iterable, lam: float = 1.0) -> "Stage12_Tensor":
    """L2 (ridge) penalty: returns scalar Tensor lam * 0.5 * sum(theta**2).
    Built from Tensor ops so grad lam*theta comes from backward(). lam==0 -> 0 Tensor."""
    return (lam * 0.5) * sum(((t ** 2).sum() for t in params)).sum()


def l1_penalty(params: Iterable, lam: float = 1.0) -> "Stage12_Tensor":
    """L1 (lasso) penalty: returns scalar Tensor lam * sum(|theta|), using _abs
    so the sub-gradient lam*sign(theta) flows through backward(). lam==0 -> 0 Tensor."""
    return lam * sum((_abs(t).sum() for t in params)).sum()


def regularized_loss(
    loss: "Stage12_Tensor",
    params: Iterable,
    *,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "Stage12_Tensor":
    """Add penalties to a stage_12 data loss: L_tilde = loss + l1*sum|theta| +
    l2*0.5*sum(theta**2). With l1==l2==0 the result must equal loss."""
    return loss + l1_penalty(params, l1) + l2_penalty(params, l2)


def l2_grad_equals_weight_decay(
    params: Iterable, lam: float
) -> List[np.ndarray]:
    """Plain-NumPy reference (no autodiff): analytic L2-penalty gradient lam*p.data
    per param, equal to stage_18's coupled weight_decay term."""
    return [lam * np.asarray(p.data, dtype=float) for p in params]
