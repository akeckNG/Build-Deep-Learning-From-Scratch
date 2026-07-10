"""Stage 22: Inverted dropout (train/eval-mode layer) on the Tensor engine (stage_08, via stage_12).

Inverted dropout with keep prob ``p``:
    train:  m ~ Bernoulli(p) elementwise, y = (m * x) / p
    eval:   y = x   (identity)
Forward is just ``x * Tensor(m / p)``, so the engine's multiply backward routes
``dL/dx = dL/dy * (m / p)``; no derivative is hand-written here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# Reuse prior stages: Tensor (stage_12 autodiff), MLP (stage_11, subclassed).
from dlfs import stage_import

Stage12_Tensor = stage_import("stage_12", "Tensor")
Stage11_MLP = stage_import("stage_11", "MLP")

# Re-export the engine so downstream stages/tests can import ``Tensor`` here.
Tensor = Stage12_Tensor


class Dropout:
    """Inverted-dropout layer with train/eval modes; forward is ``x * Tensor(m / p_keep)``.
    p_keep in (0, 1] (1.0 is identity); seed drives the private mask RNG."""

    def __init__(self, p_keep: float = 0.5, *, seed: Optional[int] = None) -> None:
        assert p_keep > 0 and p_keep <= 1
        self.p_keep = p_keep
        self._rng = np.random.default_rng(seed=seed)
        self.training = True
        self.mask = None

    def __call__(self, x) -> "Stage12_Tensor":
        """Forward: train -> ``x * Tensor(m / p_keep)`` (store scale in self.mask);
        eval -> identity."""
        if self.training:
            batched = x.data.ndim > 1
            self.mask = self._rng.binomial(1, self.p_keep, size=(x.shape[1:] if batched else x.shape)) / self.p_keep
            return x * Tensor(self.mask)
        else:
            return x

    def forward(self, x) -> "Stage12_Tensor":
        """Alias for :meth:`__call__`."""
        return self(x)

    def train(self, mode: bool = True) -> "Dropout":
        """Set ``self.training = mode`` (True -> sample + scale, False -> eval).
        Returns ``self``."""
        self.training = mode
        return self

    def eval(self) -> "Dropout":
        """Switch to eval mode (identity). Returns ``self``."""
        return self.train(False)

    def parameters(self) -> List["Stage12_Tensor"]:
        """Dropout has no learnable parameters (returns [])."""
        return []

    def zero_grad(self) -> None:
        """No parameters -> nothing to clear."""
        # TODO: implement
        raise NotImplementedError("Dropout.zero_grad")

    def __repr__(self) -> str:
        p_keep = self.p_keep
        training = self.training
        return f"Dropout({p_keep=}, {training=})"


class MLPDropout(Stage11_MLP):
    """An ``MLP`` (stage_11) with a ``Dropout`` after each hidden activation (none
    after output). Adds hidden dropouts + a training flag flipping all of them.
    activation/out_activation in ``{"tanh", "relu", "none"}``."""

    def __init__(
        self,
        sizes: Sequence[int],
        *,
        p_keep: float = 0.5,
        activation: str = "tanh",
        out_activation: str = "none",
        seed: Optional[int] = None,
    ) -> None:
        # Defer Dense-stack build + validation + size/activation storage to MLP ctor
        # (positional: sizes, activation, out_activation, seed).
        super().__init__(sizes=sizes, activation=activation, out_activation=out_activation, seed=seed)
        rng = np.random.default_rng(seed=seed)
        self.dropouts = [Dropout(p_keep=p_keep, seed=rng.integers(2**32)) for _ in range(max(0, len(sizes) - 2))]

    @property
    def training(self):
        return all(x.training for x in self.dropouts)

    def forward(self, x) -> "Stage12_Tensor":
        """Per hidden layer: dense -> activation -> dropout; output: dense ->
        out_activation (no dropout)."""
        assert isinstance(x, Tensor)

        for i, l in enumerate(self.layers):
            x = l(x)
            x = Stage11_MLP._apply_activation(x, self.activation)
            if i < len(self.layers) - 1:
                x = self.dropouts[i](x)

        return Stage11_MLP._apply_activation(x, self.out_activation)

    def train(self, mode: bool = True) -> "MLPDropout":
        """Set ``self.training = mode`` on the model AND every owned Dropout.
        Returns self."""
        for d in self.dropouts:
            d.train(mode=mode)
        return self

    def eval(self) -> "MLPDropout":
        """Put the model AND every owned Dropout in eval mode. Returns self."""
        for d in self.dropouts:
            d.eval()
        return self

    # parameters() and zero_grad() are inherited from stage_11 MLP; do not override.

    def __repr__(self) -> str:
        # TODO: short repr with sizes, p_keep, activation, out_activation.
        raise NotImplementedError("MLPDropout.__repr__")
