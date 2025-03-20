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


def interleave(
    a: UInt8Array2D,
    b: UInt8Array2D,
    step_a: int,
    step_b: int,
    start_b: int = 0,
) -> UInt8Array2D:
    """Interleaves two given 2D Uint8 arrays with given steps."""
    step = step_a + step_b
    m_a, n = a.shape
    m_b, _n = b.shape
    assert _n == n, "Arrays must have the same number of columns"
    assert 1 <= step_a <= m_a, f"Invalid {step_a = }"
    assert 1 <= step_b <= m_b, f"Invalid {step_b = }"
    assert 0 <= start_b <= m_b, f"Invalid {start_b = }"
    m = m_a + m_b
    res = np.zeros((m, n), dtype=np.uint8)
    res_to_return = res
    if start_b > 0:
        res[:start_b] = b[:start_b]
        res = res[start_b:]
        b = b[start_b:]
        m_b -= start_b
    d_a = m_a // step_a
    d_b = m_b // step_b
    d = min(d_a, d_b)
    _res = res[: d * step].reshape(d, n * step)
    _res[:, : n * step_a] = a[: d * step_a, :].reshape(d, n * step_a)
    _res[:, n * step_a :] = b[: d * step_b, :].reshape(d, n * step_b)
    r_a = m_a - d * step_a
    r_b = m_b - d * step_b
    if r_b == 0 and r_a == 0:
        return res_to_return
    if r_b == 0:
        res[-r_a:, :] = a[-r_a:, :]
    elif r_a == 0:
        res[-r_b:, :] = b[-r_b:, :]
    elif r_a <= step_a:
        res[-r_a - r_b : -r_b, :] = a[-r_a:, :]
        res[-r_b:, :] = b[-r_b:, :]
    else:
        res[-r_a - r_b : -r_a - r_b + step_a, :] = a[-r_a : -r_a + step_a, :]
        res[-r_a - r_b + step_a : -r_a + step_a, :] = b[-r_b:, :]
        res[-r_a + step_a :, :] = a[-r_a + step_a :, :]
    return res_to_return
