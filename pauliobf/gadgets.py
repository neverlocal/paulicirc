"""Pauli gadgets."""

from __future__ import annotations
from fractions import Fraction
from typing import (
    Final,
    Literal,
    Self,
    Sequence,
    TypeAlias,
    final,
)
import numpy as np

from ._numpy import (
    UInt8Array1D,
    numba_jit,
)

if __debug__:
    from typing_validation import validate


Pauli: TypeAlias = Literal[0b00, 0b01, 0b10, 0b11]
"""
Type alias for a Pauli, encoded as a 2-bit integer:
0b00 is I, 0b01 is X, 0b10 is Z, 0b11 is Y.
"""

PauliChar: TypeAlias = Literal["_", "X", "Z", "Y"]
"""
Type alias for single-character representations of Paulis.
Note that I is represented as ``_``,
following the same convention as `stim <https://github.com/quantumlib/stim>`_.
"""

PauliArray: TypeAlias = UInt8Array1D
"""
Type alias for a 1D array of Paulis, as 1D UInt8 array with entries in ``range(4)``.
"""

GadgetData: TypeAlias = UInt8Array1D
"""
Type alias for data encoding a single Pauli gadget.
This is a 1D array of bytes, where the last 2 bytes encode the phase.

The leg bit-pairs are packed, with 4 legs stored in each byte; see :obj:`Pauli`.

The phase is stored as a 16-bit integer, with the most significant byte first;
see :obj:`Phase`.
"""

Phase: TypeAlias = int
r"""
Type alias for a phase, represented as the integer :math:`k` such that the phase
angle is equal to :math:`k\pi/32768` (see :obj:`PHASE_DENOM`).
"""

PAULI_CHARS: Final[Sequence[PauliChar]] = ("_", "X", "Z", "Y")
"""
Single-character representations of Paulis,
in order compatible with the chosen encoding (cf. :obj:`Pauli`).
"""

PHASE_NBYTES: Final[Literal[2]] = 2
"""
Number of bytes used for phase representation, currently 2B  (see :obj:`PHASE_DENOM`).
"""

PHASE_DENOM: Final[int] = 256**PHASE_NBYTES
r"""
The subdivision of :math:`2\pi` used for phases.
Currently set to 65536, so that phases are integer multiples of :math:`\pi/32768`.
"""


_LEG_BYTE_SHIFTS = np.arange(6, -1, -2, dtype=np.uint8)
"""Bit shifts ``[6, 4, 2, 0]`` used on a byte to extract leg information."""

_LEG_BYTE_MASKS = 0b11 * np.ones(4, dtype=np.uint8)
"""Byte mask used on a byte to extract leg information."""


def _get_gadget_legs(g: GadgetData) -> PauliArray:
    """
    Extract an array of leg information from given gadget data.
    The returned array has values in ``range(4)``,
    where the encoding is explained in :obj:`GadgetData`.
    """
    leg_bytes = g[:-PHASE_NBYTES]
    n = len(leg_bytes)
    return (
        leg_bytes.repeat(4) & np.tile(_LEG_BYTE_MASKS << _LEG_BYTE_SHIFTS, n)
    ) >> np.tile(_LEG_BYTE_SHIFTS, n)


def _set_gadget_legs(g: GadgetData, legs: PauliArray) -> None:
    """
    Sets leg information in the given gadget data.
    The input array should have values in ``range(4)``,
    where the encoding is explained in :obj:`GadgetData`.
    """
    n = len(legs)
    leg_data = g[:-PHASE_NBYTES]
    leg_data[:] = 0
    leg_data[: -(-(n - 0) // 4)] |= legs[0::4] << 6  # type: ignore
    leg_data[: -(-(n - 1) // 4)] |= legs[1::4] << 4  # type: ignore
    leg_data[: -(-(n - 2) // 4)] |= legs[2::4] << 2  # type: ignore
    leg_data[: -(-(n - 3) // 4)] |= legs[3::4]


@numba_jit
def phase2float(phase: Phase) -> float:
    """Converts a phase (as an int) to radians (as a float)."""
    return 2 * np.pi * phase / PHASE_DENOM


@numba_jit
def float2phase(phase_f: float) -> Phase:
    """Converts radians (as a float) to a phase (as an int)."""
    return int(np.round(phase_f * PHASE_DENOM * 0.5 / np.pi)) % PHASE_DENOM


@numba_jit
def overlap(p: GadgetData, q: GadgetData) -> int:
    """Gadget overlap."""
    p = p[:-PHASE_NBYTES]
    q = q[:-PHASE_NBYTES]
    parity = np.zeros(len(p), dtype=np.uint8)
    mask = 0b00000011
    for _ in range(4):
        _p = p & mask
        _q = q & mask
        parity += (_p != 0) & (_q != 0) & (_p != _q)
        mask <<= 2
    return int(np.sum(parity))


assert (
    PHASE_NBYTES == 2
), "Functions below are implemented under the assumption of 16-bit phases."


@numba_jit
def get_phase(g: GadgetData) -> Phase:
    """Extracts phase data from the given gadget data."""
    return int(g[-2]) * 256 + int(g[-1])


@numba_jit
def set_phase(g: GadgetData, phase: Phase) -> None:
    """Sets phase data in the given gadget data."""
    g[-2], g[-1] = divmod(phase, 256)


@final
class Gadget:
    """A Pauli gadget."""

    @staticmethod
    def float2phase(rad: float) -> Phase:
        """Converts a phase (as an int) to radians (as a float)."""
        return float2phase(rad)

    @staticmethod
    def phase2float(phase: Phase) -> float:
        """Converts radians (as a float) to a phase (as an int)."""
        return phase2float(phase)

    @staticmethod
    def phase2frac(phase: Phase) -> Fraction:
        r"""Converts a phase (as an int) to a fraction of :math:`\pi`."""
        return Fraction(phase, PHASE_DENOM // 2)

    @staticmethod
    def frac2phase(frac: Fraction) -> Phase:
        r"""Converts a fraction of :math:`\pi` to a phase (as an int)."""
        assert validate(frac, Fraction)
        num, den = frac.numerator, frac.denominator
        if (PHASE_DENOM // 2) % den == 0:
            return num * PHASE_DENOM // 2 // den
        return float2phase(num / den)

    _data: GadgetData
    _num_qubits: int
    _ephemeral: bool

    def __new__(
        cls,
        data: GadgetData,
        num_qubits: int | None = None,
        *,
        _ephemeral: bool = False,
    ) -> Self:
        """Constructs a Pauli gadget from the given data."""
        assert Gadget._validate_new_args(data, num_qubits)
        if num_qubits is None:
            num_qubits = (data.shape[0] - PHASE_NBYTES) * 4
        self = super().__new__(cls)
        self._data = data
        self._num_qubits = num_qubits
        self._ephemeral = _ephemeral
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the gadget."""
        return self._num_qubits

    @property
    def legs(self) -> PauliArray:
        """Legs of the gadget."""
        return _get_gadget_legs(self._data)[: self._num_qubits]

    @legs.setter
    def legs(self, value: Sequence[Pauli] | PauliArray) -> None:
        """Sets the legs of the gadget."""
        assert validate(value, Sequence[Pauli] | PauliArray)
        legs: PauliArray = np.asarray(value, dtype=np.uint8)
        assert self._validate_legs_value(legs)
        _set_gadget_legs(self._data, legs)

    @property
    def leg_paulistr(self) -> str:
        """Paulistring representation of the gadget legs."""
        return "".join(PAULI_CHARS[p] for p in self.legs)

    @property
    def phase(self) -> Phase:
        """Phase of the gadget, as an integer; see :obj:`Phase`."""
        return get_phase(self._data)

    @phase.setter
    def phase(self, value: Phase | float | Fraction) -> None:
        r"""
        Sets the phase of the gadget:

        - exactly, as an integer;
        - approximately, as a float, rounded to the nearest multiple of :math:`\pi/32768`;
        - exactly, as a fraction with denominator dividing 32768;
        - approximately, as a fraction with any other denominator, converted to float.

        """
        if isinstance(value, Phase):
            set_phase(self._data, value)
            return
        if isinstance(value, float):
            set_phase(self._data, Gadget.float2phase(value))
            return
        set_phase(self._data, Gadget.frac2phase(value))

    @property
    def phase_float(self) -> float:
        r"""
        Approximate representation of the gadget phase,
        as a floating point number :math:`0 \leq x < \pi/32768`.
        """
        return Gadget.phase2float(self.phase)

    @property
    def phase_frac(self) -> Fraction:
        r"""
        Exact representation of the gadget phase as a fraction of :math:`\pi`.
        """
        return Gadget.phase2frac(self.phase)

    @property
    def phase_str(self) -> str:
        r"""
        String representation of the gadget phase, as a fraction of :math:`\pi`.
        """
        phase_frac = self.phase_frac
        return f"{phase_frac.numerator}π/{phase_frac.denominator}"

    def clone(self) -> Self:
        """Creates a persistent copy of the gadget."""
        return Gadget(self._data.copy(), self._num_qubits)

    def overlap(self, other: Gadget) -> int:
        """
        Returns the overlap between the legs of this gadgets and those of the given
        gadget, computed as the number of qubits where the legs of the two gadgets
        differ and are both not the identity Pauli (the value 0, as a :obj:`Pauli`).
        """
        assert self._validate_same_num_qubits(other)
        return overlap(self._data, other._data)

    def __repr__(self) -> str:
        legs_str = self.leg_paulistr
        if len(legs_str) > 16:
            legs_str = legs_str[:8] + "..." + legs_str[-8:]
        return f"<Gadget: {legs_str}, {self.phase_str}>"

    if __debug__:

        @staticmethod
        def _validate_new_args(
            data: GadgetData, num_qubits: int | None
        ) -> Literal[True]:
            """Validates the arguments of the :meth:`__new__` method."""
            validate(data, GadgetData)
            if num_qubits is not None:
                validate(num_qubits, int)
                if num_qubits < 0:
                    raise ValueError("Number of qubits must be non-negative.")
                if num_qubits > data.shape[0] * 4:
                    raise ValueError("Number of qubits exceeds circuit width.")
            return True

        def _validate_legs_value(self, legs: PauliArray) -> Literal[True]:
            """Validates the value of the :attr:`legs` property."""
            if len(legs) != self.num_qubits:
                raise ValueError("Number of legs does not match number of qubits.")
            if not all(0 <= leg < 4 for leg in legs):
                raise ValueError("Legs must have value in range(4).")
            return True

        def _validate_same_num_qubits(self, gadget: Gadget) -> Literal[True]:
            validate(gadget, Gadget)
            if self.num_qubits != gadget.num_qubits:
                raise ValueError("Mismatch in number of qubits between gadgets.")
            return True
