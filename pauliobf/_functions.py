"""Low-level implementation of Pauli circuit operations."""

from __future__ import annotations
from collections.abc import Callable
from typing import Final, Literal, ParamSpec, TypeAlias, TypeVar
import numpy as np
import numba # type: ignore

_numba_jit = numba.jit(nopython=True, cache=True)
"""Decorator to apply :func:`numba.jit` with desired settings."""

_P = ParamSpec("_P")
"""Type alias for a generic parameter list."""

_R = TypeVar("_R")
"""Type alias for a generic return type."""

def numba_jit(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator to apply :func:`numba.jit` with desired settings."""
    return _numba_jit(func) # type: ignore

UInt8Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint8]]
"""Type alias for 1D UInt8 NumPy arrays."""

UInt8Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
"""Type alias for 2D UInt8 NumPy arrays."""

GadgetData: TypeAlias = UInt8Array1D
"""
Type alias for data encoding a single Pauli gadget.
This is a 1D array of bytes, where the last 2 bytes encode the phase.

The leg bit-pairs are packed, with 4 legs stored in each byte.
We use the following 2-bit encoding for the Paulis:
0b00 is I, 0b01 is X, 0b10 is Z, 0b11 is Y.

The phase is stored as a 16-bit integer, with the most significant byte first;
see :obj:`Phase`.
"""

GadgetCircData: TypeAlias = UInt8Array2D
"""Type alias for data encoding a circuit of Pauli gadgets."""

PHASE_NBYTES: Final[Literal[2]] = 2
"""
Number of bytes used for phase representation, currently 2B  (see :obj:`PHASE_DENOM`).
"""

PHASE_DENOM: Final[int] = 256**PHASE_NBYTES
r"""
The subdivision of :math:`2\pi` used for phases.
Currently set to 65536, so that phases are integer multiples of :math:`\pi/32768`.
"""

PhaseDataArray: TypeAlias = np.ndarray[tuple[int, Literal[2]], np.dtype[np.uint8]]
"""
Type alias for a 1D array of encoded phase data,
as a 2D UInt8 NumPy array of shape ``(n, 2)``, where ``n`` is the number of phases.
"""

Phase: TypeAlias = int
"""
Type alias for a phase, represented as the integer :math:`k` such that the phase
angle is equal to :math:`k\pi/32768` (see :obj:`PHASE_DENOM`).
"""

PhaseArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
"""Type alias for a 1D array of phases, as a 1D UInt16 NumPy array."""

RNG: TypeAlias = np.random.Generator
"""Typa alias for a NumPy random number generator."""

assert PHASE_NBYTES == PhaseDataArray.__args__[0].__args__[1].__args__[0] # type: ignore
assert PhaseArray.__args__[1].__args__[0].__name__ == f"uint{8*PHASE_NBYTES}" # type: ignore

def circ_ncols(m: int, n: int) -> int:
    """
    Number of columns for a circuit with ``m`` gadgets on ``n`` qubits.

    :raises NotImplementedError: if the number ``n`` of qubits is not divisible by 4.
    """
    if n%4 != 0:
        raise NotImplementedError("Number of qubits must be divisible by 4.")
    legs_nbytes = -(-n//4)
    return legs_nbytes+PHASE_NBYTES

def shape(circ: GadgetCircData) -> tuple[int, int]:
    """
    Given gadget circuit data, returns the pair ``(m,n)`` of
    the number ``m`` of gadgets and the number ``n`` of qubits.
    """
    m, _n = circ.shape
    n = (_n-PHASE_NBYTES)*4
    return m, n

def get_phases(circ: GadgetCircData) -> PhaseDataArray:
    """Returns the array of phases for the gadgets in a given circuit."""
    return circ[:, -PHASE_NBYTES:]

def set_phases(circ: GadgetCircData, phases: PhaseDataArray) -> None:
    """Sets the array of phases for the gadgets in the given circuit."""
    circ[:, -PHASE_NBYTES:] = phases

def zero_circ(m: int, n: int) -> GadgetCircData:
    """
    Returns a circuit with ``m`` gadgets on ``n`` qubits,
    where all gadgets have no legs and zero phase.

    :raises NotImplementedError: if the number ``n`` of qubits is not divisible by 4.
    """
    return np.zeros((m, circ_ncols(m, n)), dtype=np.uint8)

def rand_circ(m: int, n: int, *, rng: RNG) -> GadgetCircData:
    """
    Returns a uniformly random circuit with ``m`` gadgets on ``n`` qubits,
    where all gadgets have no legs and zero phase.

    :raises NotImplementedError: if the number ``n`` of qubits is not divisible by 4.
    """
    return rng.integers(0, 256, (m, circ_ncols(m, n)), dtype=np.uint8)

_LEG_BYTE_SHIFTS = np.arange(6, -1, -2, dtype=np.uint8)
"""Bit shifts ``[6, 4, 2, 0]`` used on a byte to extract leg information."""

_LEG_BYTE_MASKS = 0b11*np.ones(4, dtype=np.uint8)
"""Byte mask used on a byte to extract leg information."""

def gadget_legs(g: GadgetData) -> UInt8Array1D:
    """
    Extract an array of leg information from given gadget data.
    The returned array has values in ``range(4)``,
    where the encoding is explained in :obj:`GadgetData`.
    """
    leg_bytes = g[:-PHASE_NBYTES]
    n = len(leg_bytes)
    return (
        leg_bytes.repeat(4)
        & np.tile(_LEG_BYTE_MASKS << _LEG_BYTE_SHIFTS, n)
    ) >> np.tile(_LEG_BYTE_SHIFTS, n)

@numba_jit
def phase2rad(phase: Phase) -> float:
    """Converts a phase (as an integer) to radians."""
    return np.pi*phase/PHASE_DENOM

@numba_jit
def rad2phase(phase_f: float) -> Phase:
    """Converts radians to a phase (as an integer)."""
    return int(np.round(phase_f*PHASE_DENOM/np.pi))%PHASE_DENOM

assert PHASE_NBYTES == 2, (
    "Functions below are implemented under the assumption of 16-bit phases."
)

@numba_jit
def get_phase(g: GadgetData) -> Phase:
    """Extracts phase data from the given gadget data."""
    return int(g[-2])*256+int(g[-1])

@numba_jit
def set_phase(g: GadgetData, phase: Phase) -> None:
    """Sets phase data in the given gadget data."""
    g[-2], g[-1] = divmod(phase, 256)
