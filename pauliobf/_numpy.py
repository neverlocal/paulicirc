"""NumPy-related utility types and functions."""

from __future__ import annotations
from collections.abc import Callable
from typing import ParamSpec, TypeAlias, TypeVar
import numpy as np
import numba  # type: ignore

UInt8Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint8]]
"""Type alias for 1D UInt8 NumPy arrays."""

UInt16Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
"""Type alias for 1D UInt16 NumPy arrays."""

UInt8Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
"""Type alias for 2D UInt8 NumPy arrays."""

RNG: TypeAlias = np.random.Generator
"""Typa alias for a NumPy random number generator."""

_numba_jit = numba.jit(nopython=True, cache=True)
"""Decorator to apply :func:`numba.jit` with desired settings."""

_P = ParamSpec("_P")
"""Type alias for a generic parameter list."""

_R = TypeVar("_R")
"""Type alias for a generic return type."""


def numba_jit(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator to apply :func:`numba.jit` with desired settings."""
    return _numba_jit(func)  # type: ignore
