"""NumPy-related utility types and functions."""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from collections.abc import Callable
from typing import Any, ParamSpec, TypeAlias, TypeVar
import numpy as np
import numba  # type: ignore

BoolArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
"""Type alias for 1D bool NumPy arrays."""

UInt8Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint8]]
"""Type alias for 1D uint8 NumPy arrays."""

UInt16Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
"""Type alias for 1D uint16 NumPy arrays."""

UInt8Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
"""Type alias for 2D uint8 NumPy arrays."""

FloatArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
"""Type alias for 1D float NumPy arrays."""

Complex128Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.complex128]]
"""Type alias for 1D complex128 NumPy arrays."""

Complex128Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Type alias for 2D complex128 NumPy arrays."""

RNG: TypeAlias = np.random.Generator
"""Typa alias for a NumPy random number generator."""

ShapeT = TypeVar("ShapeT", bound=tuple[int, ...])
"""Type variable for the shape of NumPy arrays."""

_numba_jit = numba.jit(nopython=True, cache=True)
"""Decorator to apply :func:`numba.jit` with desired settings."""

_P = ParamSpec("_P")
"""Type alias for a generic parameter list."""

_R = TypeVar("_R")
"""Type alias for a generic return type."""


def numba_jit(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator to apply :func:`numba.jit` with desired settings."""
    return _numba_jit(func)  # type: ignore

def normalise_phase(
    array: np.ndarray[ShapeT, np.dtype[np.complex128]],
    *,
    tol: float = 1e-8,
) -> None:
    """Normalises the phase of a complex array (in place)."""
    assert tol > 0
    idx = np.argmax(abs(array) >= tol)
    val = array.flatten()[idx]
    if (aval := abs(val)) >= tol:
        array *= aval / val
