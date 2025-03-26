"""Pauli gadgets."""

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
from collections.abc import Iterator
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
from scipy.linalg import expm  # type: ignore[import-untyped]

from ._numpy import (
    Complex128Array1D,
    Complex128Array2D,
    UInt8Array1D,
    normalise_phase,
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

PAULI_MATS: Final[tuple[Complex128Array2D, ...]] = (
    np.array([[1, 0], [0, 1]], dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
)
"""The four Pauli matrices."""

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


def _zero_gadget_data(num_qubits: int) -> GadgetData:
    """Returns blank data for a gadget with the given number of qubits."""
    return np.zeros(-(-num_qubits // 4) + PHASE_NBYTES, dtype=np.uint8)


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

    @staticmethod
    def assemble_data(legs: PauliArray, phase: Phase) -> GadgetData:
        """Assembles gadget data from the given legs and phase."""
        g = _zero_gadget_data(len(legs))
        _set_gadget_legs(g, legs)
        set_phase(g, phase)
        return g

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
        as a floating point number :math:`0 \leq x < 2\pi`.
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
        num, den = phase_frac.numerator, phase_frac.denominator
        if num == 0:
            return "0"
        num_str = "" if num == 1 else str(num)
        if den == 1:
            return f"{num_str}π"  # the only case should be 'π'
        return f"{num_str}π/{str(den)}"

    def unitary(self, *, _normalise_phase: bool = True) -> Complex128Array2D:
        """Returns the unitary matrix associated to this Pauli gadget."""
        legs = self.legs
        kron_prod = PAULI_MATS[legs[0]]
        for leg in legs[1:]:
            kron_prod = np.kron(kron_prod, PAULI_MATS[leg])
        res: Complex128Array2D = expm(-0.5j * self.phase_float * kron_prod)
        if _normalise_phase:
            normalise_phase(res)
        return res

    def statevec(
        self, input: Complex128Array1D, _normalise_phase: bool = False
    ) -> Complex128Array1D:
        """
        Computes the statevector resulting from the application of this gadget
        to the given input statevector.
        """
        assert validate(input, Complex128Array1D)
        res = self.unitary(_normalise_phase=False) @ input
        if _normalise_phase:
            normalise_phase(res)
        return res

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
            if num_qubits is None:
                num_qubits = 4*(len(data)-PHASE_NBYTES)
            else:
                validate(num_qubits, int)
                if num_qubits < 0:
                    raise ValueError("Number of qubits must be non-negative.")
                if num_qubits > (data.shape[0] - PHASE_NBYTES) * 4:
                    raise ValueError("Number of qubits exceeds circuit width.")
            legs = _get_gadget_legs(data)
            if any(legs[num_qubits:] != 0):
                raise ValueError("Legs on excess qubits must be zeroed out.")
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


class Layer:
    """A layer of Pauli gadgets with compatible legs."""

    @staticmethod
    def _legs_to_subset(legs: PauliArray) -> int:
        """Convert legs to a subset index."""
        subset = 0
        for i, leg in enumerate(legs):
            if leg != 0:
                subset |= 1 << i
        return subset

    @staticmethod
    def _subset_to_legs(subset: int, legs: PauliArray) -> PauliArray:
        return np.where(
            np.fromiter((subset & (1 << x) for x in range(len(legs))), dtype=np.bool_),
            legs,
            0,
        )

    _phases: dict[int, Phase]
    _legs: PauliArray
    _leg_count: np.ndarray[tuple[int], np.dtype[np.uint32]]
    # FIXME: remove limit to 2**32 gadgets per layer

    def __new__(cls, num_qubits: int) -> Self:
        """
        Create an empty Pauli layer with the given number of qubits.

        :meta public:
        """
        assert Layer._validate_new_args(num_qubits)
        self = super().__new__(cls)
        self._phases = {}
        self._legs = np.zeros(num_qubits, dtype=np.uint8)
        self._leg_count = np.zeros(num_qubits, dtype=np.uint32)
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits for the Pauli layer."""
        return len(self._legs)

    @property
    def legs(self) -> PauliArray:
        """Legs of the Pauli layer."""
        view = self._legs.view()
        view.setflags(write=False)
        return view.view()

    @property
    def leg_paulistr(self) -> str:
        """Paulistring representation of the layer's legs."""
        return "".join(PAULI_CHARS[p] for p in self.legs)

    def phase(self, legs: PauliArray) -> Phase | None:
        """
        Get the phase of the given legs in the layer, or :obj:`None` if the legs
        are incompatible with the layer.
        """
        if not self.is_compatible_with(legs):
            return None
        return self._phases.get(Layer._legs_to_subset(legs), 0)

    def is_compatible_with(self, legs: PauliArray) -> bool:
        """Check if the legs are compatible with the current layer."""
        assert self._validate_legs(legs)
        self_legs = self._legs
        return bool(np.all((self_legs == legs) | (self_legs == 0) | (legs == 0)))

    def commutes_with(self, legs: PauliArray) -> bool:
        """Check if the legs commute with the current layer."""
        assert self._validate_legs(legs)
        self_legs = self._legs
        for subset in self._phases:
            subset_legs = Layer._subset_to_legs(subset, self_legs)
            ovlp = sum((subset_legs != legs) & (subset_legs != 0) & (subset_legs != 0))
            if ovlp % 2 != 0:
                return False
        return True

    def add_gadget(self, legs: PauliArray, phase: Phase) -> bool:
        """Add a gadget to the layer."""
        if not self.is_compatible_with(legs):
            return False
        phase %= PHASE_DENOM
        if phase == 0:
            return True
        phases = self._phases
        subset = Layer._legs_to_subset(legs)
        if subset in phases:
            new_phase = (phases[subset] + phase) % PHASE_DENOM
            if new_phase == 0:
                # print("add_gadget", legs, phase)
                # print(list(self))
                # print(f"{subset:0{self.num_qubits}b}")
                # print(phases[subset], phase, new_phase)
                # print(self._legs)
                # print(self._leg_count)
                del phases[subset]
                self._leg_count -= np.where(legs == 0, np.uint32(0), np.uint32(1))
                self._legs = np.where(self._leg_count == 0, 0, self._legs)
                # print(self._legs)
                # print(self._leg_count)
                # print(list(self))
                # print()
            else:
                phases[subset] = new_phase
            return True
        else:
            phases[subset] = phase
            self._leg_count += np.where(legs == 0, np.uint32(0), np.uint32(1))
            self._legs = np.where(legs == 0, self._legs, legs)
        return True

    def __iter__(self) -> Iterator[Gadget]:
        """
        Iterates over the gadgets in the layer, in insertion order.

        :meta public:
        """
        legs, num_qubits = self._legs, self.num_qubits
        for subset, phase in self._phases.items():
            subset_legs = Layer._subset_to_legs(subset, legs)
            yield Gadget(Gadget.assemble_data(subset_legs, phase), num_qubits)

    def __len__(self) -> int:
        """The number of gadgets (with non-zero phase) in this layer."""
        return len(self._phases)

    def __repr__(self) -> str:
        legs_str = self.leg_paulistr
        if len(legs_str) > 16:
            legs_str = legs_str[:8] + "..." + legs_str[-8:]
        return f"<Layer: {legs_str}, {len(self)} gadgets>"

    if __debug__:

        @staticmethod
        def _validate_new_args(num_qubits: int) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, int)
            if num_qubits <= 0:
                raise ValueError("Number of qubits must be strictly positive.")
            return True

        def _validate_legs(self, legs: PauliArray) -> Literal[True]:
            """Validate gadget legs for use with this layer."""
            validate(legs, PauliArray)
            if len(legs) != self.num_qubits:
                raise ValueError(
                    "Legs must have the same length as the number of qubits."
                )
            return True
