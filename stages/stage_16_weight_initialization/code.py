"""Stage 16: Weight initialization.

Turns the variance-propagation rule Var(z) = n_in * Var(W) * Var(x) into the
Xavier/Glorot (tanh/linear) and He/Kaiming (relu) schemes, plus `init_dense`
(apply an init to a Dense in place) and the `forward_activation_stats` harness
that measures activation statistics across depth. Imports Dense (stage_10) and
Tensor (stage_08) as-is; nothing is redefined.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Dense (stage_10) and Tensor (stage_08) via the shared dlfs shim, used as-is.
from dlfs import stage_import

Stage10_Dense = stage_import("stage_10", "Dense")
Stage8_Tensor = stage_import("stage_08", "Tensor")

Dense = Stage10_Dense
Tensor = Stage8_Tensor


def xavier_uniform(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot uniform init: U[-a, a], a = gain*sqrt(6/(n_in+n_out)). Returns (n_in, n_out)."""
    a = gain * np.sqrt(6.0 / (n_in + n_out))
    return np.random.default_rng(seed).uniform(-a, a, size=(n_in, n_out))


def xavier_normal(
    n_in: int, n_out: int, *, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """Xavier/Glorot normal init: N(0, std**2), std = gain*sqrt(2/(n_in+n_out)). Returns (n_in, n_out)."""
    std = gain * np.sqrt(2.0 / (n_in + n_out))
    return np.random.default_rng(seed).normal(0.0, std, size=(n_in, n_out))


def he_normal(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming normal init for ReLU nets: N(0, 2/n_in). Returns (n_in, n_out)."""
    std = np.sqrt(2.0 / n_in)
    return np.random.default_rng(seed).normal(0.0, std, size=(n_in, n_out))


def he_uniform(n_in: int, n_out: int, *, seed: Optional[int] = None) -> np.ndarray:
    """He/Kaiming uniform init for ReLU nets: U[-a, a], a = sqrt(6/n_in). Returns (n_in, n_out)."""
    a = np.sqrt(6.0 / n_in)
    return np.random.default_rng(seed).uniform(-a, a, size=(n_in, n_out))


def init_dense(layer, W: np.ndarray, b: Optional[np.ndarray] = None) -> None:
    """Overwrite a stage_10 ``Dense`` layer's weights IN PLACE with `W` (and `b`).

    Keep ``layer.W`` (and ``layer.b``) the SAME leaf ``Tensor`` objects — only
    their ``.data`` is replaced — so any optimizer/graph reference keeps
    accumulating into the same ``.grad`` buffers. Reset those ``.grad``s to
    zeros of the new shape. Raise ``ValueError`` on any shape mismatch.
    """
    W = np.asarray(W, dtype=np.float64)
    current = np.asarray(layer.W.data)
    if W.shape != current.shape:
        raise ValueError(f"init_dense: W shape {W.shape} != layer.W shape {current.shape}")
    layer.W.data = W.copy()
    layer.W.grad = np.zeros_like(W)

    if b is not None:
        if layer.b is None:
            raise ValueError("init_dense: layer has no bias but b was given")
        b = np.asarray(b, dtype=np.float64)
        target = np.asarray(layer.b.data)
        if b.size != target.size:
            raise ValueError(f"init_dense: b size {b.size} != layer.b size {target.size}")
        layer.b.data = b.reshape(target.shape).copy()
        layer.b.grad = np.zeros_like(target)


def forward_activation_stats(
    sizes: Sequence[int],
    init_fn: Callable[[int, int], np.ndarray],
    activation: str,
    *,
    n_samples: int = 512,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Push N(0,1) inputs through a freshly initialized Dense stack; record stats.

    Build ``len(sizes)-1`` stage_10 ``Dense`` layers (layer k maps
    ``sizes[k] -> sizes[k+1]``), init each weight with ``init_fn(n_in, n_out)``
    and zero bias (via ``init_dense``), then push ``n_samples`` standard-normal
    rows through the stack, applying ``activation`` ("tanh" / "relu" / "none")
    after every layer. Return one dict per layer with keys:

      - "mean":      mean of that layer's post-activation output
      - "std":       std of that layer's post-activation output
      - "saturated": fraction with |value| > 0.98 (tanh saturation measure)
      - "dead":      fraction exactly 0.0 (dead-ReLU measure)

    This is the measurement harness that shows WHY the variance-matched inits
    matter: matched init keeps "std" stable across depth; a tiny init collapses
    it toward 0; a large one saturates tanh / kills ReLUs.
    """
    if activation not in ("tanh", "relu", "none"):
        raise ValueError(f"unknown activation {activation!r}")

    rng = np.random.default_rng(seed)
    layers = []
    for k in range(len(sizes) - 1):
        n_in, n_out = sizes[k], sizes[k + 1]
        layer = Dense(n_in, n_out, bias=True, seed=None if seed is None else seed + k)
        init_dense(layer, init_fn(n_in, n_out), b=np.zeros(n_out))
        layers.append(layer)

    h = Tensor(rng.standard_normal((n_samples, sizes[0])))
    stats: List[Dict[str, float]] = []
    for layer in layers:
        z = layer(h)
        if activation == "tanh":
            h = z.tanh()
        elif activation == "relu":
            h = z.relu()
        else:
            h = z
        d = np.asarray(h.data)
        stats.append(
            {
                "mean": float(d.mean()),
                "std": float(d.std()),
                "saturated": float((np.abs(d) > 0.98).mean()),
                "dead": float((d == 0.0).mean()),
            }
        )
    return stats
