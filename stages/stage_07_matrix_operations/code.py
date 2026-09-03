"""Stage 07: Matrix operations.

A `Mat` (2-D matrix of `Value` scalars) on top of the stage_05 scalar engine;
gradients flow through `Value`. For C = A @ B: dL/dA = G @ B.T, dL/dB = A.T @ G.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from typing import Iterable, List, Tuple, Union

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from dlfs import stage_import  # noqa: E402

# Reuse the scalar engine (stage_05) and vector container (stage_06) unchanged.
Stage5_Value = stage_import("stage_05", "Value")
Stage6_Vec = stage_import("stage_06", "Vec")

# Re-export under canonical public names for later stages / tests.
Value = Stage5_Value
Vec = Stage6_Vec

Scalar = Union[int, float, Stage5_Value]


class Mat:
    """A 2-D matrix of `Value` scalars (row-major `List[List[Value]]`). Ops compose
    scalar `Value` arithmetic; no `backward()` (reduce to a scalar first). Each op
    returns a NEW `Mat`."""

    def __init__(self, data: Iterable[Iterable[Scalar]]) -> None:
        """Build self.data: List[List[Value]] from a 2-D iterable; set rows/cols
        (wrap non-`Value` entries, keep shared nodes; ValueError on ragged rows)."""
        self.data: List[List[Stage5_Value]] = [
                [
                    j if isinstance(j, Value) else Value(j)
                    for j in i
                ] 
                for i in data
            ]
        self.rows: int = len(self.data)
        self.cols: int = len(self.data[0])
        for i in self:
            assert len(i) == self.cols

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, i: int) -> List[Stage5_Value]:
        """Return row `i` as a list of `Value`."""
        return self.data[i]

    def __iter__(self):
        """Iterate over rows."""
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Mat(shape={self.shape})"

    def matmul(self, other: "Mat") -> "Mat":
        """Matrix product C = self @ other; (m,k)@(k,n) -> (m,n), where
        C[i][j] = sum_p self[i][p] * other[p][j]. ValueError on inner-dim mismatch."""
        assert self.cols == other.rows
        C = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for p in range(self.cols):
                    C[i][j] += self[i][p] * other[p][j]

        return type(self)(C)

    def __matmul__(self, other: "Mat") -> "Mat":
        """The `@` operator; delegates to `matmul`."""
        return self.matmul(other)

    def transpose(self) -> "Mat":
        """Transpose: C[j][i] = self[i][j], shape (cols, rows); reuse the same
        `Value` objects so gradient flows back to the originals."""
        C = [[None for _ in range(self.rows)] for _ in range(self.cols)]
        for i in range(self.rows):
            for j in range(self.cols):
                C[j][i] = self[i][j]
        return Mat(C)

    @property
    def T(self) -> "Mat":
        """Property alias for `transpose()`."""
        return self.transpose()

    def reshape(self, rows: int, cols: int) -> "Mat":
        """Reshape to (rows, cols) row-major, reusing the `Value`s; ValueError if
        the element count changes."""
        if self.rows * self.cols != rows * cols:
            raise ValueError
        iC = 0
        jC = 0
        C = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(self.rows):
            for j in range(self.cols):
              C[iC][jC] = self[i][j]
              jC+=1
              if (jC >= cols):
                iC+=1
                jC=0
        return Mat(C)


    def sum(self) -> Stage5_Value:
        """Sum of all elements as one `Value`: s = sum_{i,j} self[i][j]."""
        return sum([sum(i) for i in self])

    def mean(self) -> Stage5_Value:
        """Mean of all elements as one `Value`: sum() / N, N = rows*cols."""
        return self.sum()/(self.cols*self.rows)
