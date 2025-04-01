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
from collections.abc import Iterable, Iterator, Set
from fractions import Fraction
import re
from typing import (
    Any,
    Final,
    Literal,
    Self,
    Sequence,
    TypeAlias,
    final,
    overload,
)
import numpy as np
from scipy.linalg import expm  # type: ignore[import-untyped]

from ._numpy import (
    RNG,
    Complex128Array1D,
    Complex128Array2D,
    FloatArray1D,
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

PauliArray: TypeAlias = UInt8Array1D
"""
Type alias for a 1D array of Paulis, as 1D UInt8 array with entries in ``range(4)``.
"""

PauliChar: TypeAlias = Literal["_", "X", "Z", "Y"]
"""
Type alias for single-character representations of Paulis.
Note that I is represented as ``_``,
following the same convention as `stim <https://github.com/quantumlib/stim>`_.
"""

PAULI_CHARS: Final[Sequence[PauliChar]] = ("_", "X", "Z", "Y")
"""
Single-character representations of Paulis,
in order compatible with the chosen encoding (cf. :obj:`Pauli`).
"""

PAULI_MATS: Final[tuple[Complex128Array2D, ...]] = (
    np.array([[1, 0], [0, 1]], dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
)
"""The four Pauli matrices."""

GadgetData: TypeAlias = UInt8Array1D
"""
Type alias for data encoding a single Pauli gadget.
This is a 1D array of bytes, where the last 2 bytes encode the phase.

The leg bit-pairs are packed, with 4 legs stored in each byte; see :obj:`Pauli`.

The phase is stored as a 16-bit integer, with the most significant byte first;
see :obj:`Phase`.
"""

Phase: TypeAlias = float
r"""Type alias for a phase, as a 64-bit float."""


PHASE_DTYPE: Final[type[np.floating[Any]]] = np.float64
"""NumPy dtype used to represent phases."""

PHASE_NBYTES: Final[int] = int(
    re.match(r"float([0-9]+)", PHASE_DTYPE.__name__)[1] # type: ignore
)//8
"""Number of bytes used for phase representation."""

assert PHASE_NBYTES >= 2, "Code presumes at least 16-bit precision."


PhaseDataArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
"""
Type alias for a 1D array of encoded phase data,
as a 2D UInt8 NumPy array of shape ``(n, PHASE_NBYTES)``.
"""

PhaseArray: TypeAlias = FloatArray1D
"""Type alias for a 1D array of phases."""

_LEG_BYTE_SHIFTS = np.arange(6, -1, -2, dtype=np.uint8)
"""Bit shifts ``[6, 4, 2, 0]`` used on a byte to extract leg information."""

_LEG_BYTE_MASKS = 0b11 * np.ones(4, dtype=np.uint8)
"""Byte mask used on a byte to extract leg information."""


def zero_gadget_data(num_qubits: int) -> GadgetData:
    """Returns blank data for a gadget with the given number of qubits."""
    return np.zeros(-(-num_qubits // 4) + PHASE_NBYTES, dtype=np.uint8)

def get_gadget_legs(g: GadgetData) -> PauliArray:
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


def set_gadget_legs(g: GadgetData, legs: PauliArray) -> None:
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
def get_phase(g: GadgetData) -> Phase:
    """Extracts phase data from the given gadget data."""
    return float(g[-PHASE_NBYTES:].view(np.float64)[0])

@numba_jit
def set_phase(g: GadgetData, phase: Phase) -> None:
    """Sets phase data in the given gadget data."""
    g[-PHASE_NBYTES:] = np.array([phase % (2*np.pi)], dtype=np.float64).view(np.uint8)

@numba_jit
def is_zero_phase(phase: Phase) -> bool:
    """Whether the given phase is deemed to be zero."""
    atol = 1e-8
    phase %= 2*np.pi
    return bool(phase < atol or 2*np.pi-phase < atol)

@numba_jit
def are_same_phase(lhs: Phase, rhs: Phase) -> bool:
    """Whether the given phases are deemed to be the same."""
    lhs %= 2*np.pi
    rhs %= 2*np.pi
    return bool(
        np.isclose(lhs, rhs)
        or np.isclose(lhs, 2*np.pi-rhs)
    )

@numba_jit
def gadget_overlap(p: GadgetData, q: GadgetData) -> int:
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

@numba_jit
def decode_phases(phase_data: PhaseDataArray) -> PhaseArray:
    """Decodes phase data from a gadget circuit into an array of phases."""
    return phase_data.flatten().view(PHASE_DTYPE)

@numba_jit
def encode_phases(phases: PhaseArray) -> PhaseDataArray:
    """Encodes an array of phases into phase data for a gadget circuit."""
    return phases.view(np.uint8).reshape(-1, PHASE_NBYTES)

@numba_jit
def invert_phases(phase_data: PhaseDataArray) -> None:
    """Inplace phase inversion for the given phase data."""
    phase_data[:] = encode_phases(-decode_phases(phase_data))

@final
class Gadget:
    """A Pauli gadget."""

    @staticmethod
    def phase2frac(phase: Phase, *, prec: int = 8) -> Fraction:
        r"""
        Converts a phase to a fraction of :math:`\pi`.

        The optional ``prec`` kwarg can be used to set a number of bits of precision;
        default is ``prec=8``, corresponding to :math:`\pi/256`.
        """
        K = 2**prec
        return Fraction(round(phase/np.pi*K)%K, K)

    @staticmethod
    def frac2phase(frac: Fraction) -> Phase:
        r"""Converts a fraction of :math:`\pi` to a phase (as a float)."""
        return float(frac)*np.pi

    @staticmethod
    def assemble_data(legs: PauliArray, phase: Phase) -> GadgetData:
        """Assembles gadget data from the given legs and phase."""
        assert Gadget._validate_legs(legs)
        g = zero_gadget_data(len(legs))
        set_gadget_legs(g, legs)
        set_phase(g, phase)
        return g

    @classmethod
    def zero(cls, num_qubits: int) -> Self:
        """Returns the gadget with no legs and zero phase."""
        data = zero_gadget_data(num_qubits)
        return cls(data, num_qubits)

    @classmethod
    def from_legs(cls, legs: PauliArray, phase: Phase) -> Self:
        """Returns the gadget with given legs and phase."""
        assert Gadget._validate_legs(legs)
        num_qubits = len(legs)
        data = Gadget.assemble_data(legs, phase)
        return cls(data, num_qubits)

    @classmethod
    def from_paulistr(cls, paulistr: str, phase: Phase) -> Self:
        """Returns the gadget with given legs (as paulistr) and phase."""
        assert Gadget._validate_paulistr(paulistr)
        num_qubits = len(paulistr)
        legs = np.fromiter((PAULI_CHARS.index(p) for p in paulistr), dtype=np.uint8)
        data = Gadget.assemble_data(legs, phase)
        return cls(data, num_qubits)

    @classmethod
    def random(
        cls,
        num_qubits: int,
        *,
        allow_zero: bool = True,
        allow_legless: bool = True,
        rng: int | RNG | None = None
        ) -> Self:
        """Returns a gadget with uniformly sampled legs and phase."""
        if not isinstance(rng, RNG):
            rng = np.random.default_rng(rng)
        legs: PauliArray = rng.integers(0, 4, size=num_qubits, dtype=np.uint8)
        if not allow_legless:
            if num_qubits == 0:
                raise ValueError("Number of qubits must be positive.")
            while np.all(legs == 0):
                legs = rng.integers(0, 4, size=num_qubits, dtype=np.uint8)
        phase: Phase = rng.uniform(0, 2*np.pi)
        if not allow_zero:
            while is_zero_phase(phase):
                phase = rng.uniform(0, 2*np.pi)
        data = Gadget.assemble_data(legs, phase)
        return cls(data, num_qubits)


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
        return get_gadget_legs(self._data)[: self._num_qubits]

    @legs.setter
    def legs(self, value: Sequence[Pauli] | PauliArray) -> None:
        """Sets the legs of the gadget."""
        assert validate(value, Sequence[Pauli] | PauliArray)
        legs: PauliArray = np.asarray(value, dtype=np.uint8)
        assert self._validate_legs_self(legs)
        set_gadget_legs(self._data, legs)

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
        set_phase(self._data, Gadget.frac2phase(value))

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

    @property
    def is_zero(self) -> bool:
        """Whether the gadget has zero phase."""
        return is_zero_phase(self.phase)

    @property
    def is_legless(self) -> bool:
        """Whether the gadget has no legs."""
        return bool(np.all(self.legs == 0))

    def overlap(self, other: Gadget) -> int:
        """
        Returns the overlap between the legs of this gadgets and those of the given
        gadget, computed as the number of qubits where the legs of the two gadgets
        differ and are both not the identity Pauli (the value 0, as a :obj:`Pauli`).
        """
        assert self._validate_same_num_qubits(other)
        return gadget_overlap(self._data, other._data)

    def unitary(self, *, _normalise_phase: bool = True) -> Complex128Array2D:
        """Returns the unitary matrix associated to this Pauli gadget."""
        legs = self.legs
        kron_prod = PAULI_MATS[legs[0]]
        for leg in legs[1:]:
            kron_prod = np.kron(kron_prod, PAULI_MATS[leg])
        res: Complex128Array2D = expm(-0.5j * self.phase * kron_prod)
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Gadget):
            return NotImplemented
        return (
            self.num_qubits == other.num_qubits
            and np.array_equal(self._data[:-PHASE_NBYTES], other._data[:-PHASE_NBYTES])
            and are_same_phase(self.phase, other.phase)
        )

    def __repr__(self) -> str:
        legs_str = self.leg_paulistr
        if len(legs_str) > 16:
            legs_str = legs_str[:8] + "..." + legs_str[-8:]
        return f"<Gadget: {legs_str}, {self.phase:.15f}>"

    if __debug__:

        @staticmethod
        def _validate_paulistr(paulistr: str) -> Literal[True]:
            """Validates a given Paulistring."""
            validate(paulistr, str)
            if not all(p in PAULI_CHARS for p in paulistr):
                raise ValueError("Paulistring characters must be '_', 'X', 'Z' or 'Y'.")
            return True

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
            legs = get_gadget_legs(data)
            if any(legs[num_qubits:] != 0):
                raise ValueError("Legs on excess qubits must be zeroed out.")
            return True

        @staticmethod
        def _validate_legs(legs: PauliArray) -> Literal[True]:
            """Validate gadget legs for use with this layer."""
            validate(legs, PauliArray)
            if not np.all(legs < 4):
                raise ValueError("Leg values must be in range(4).")
            return True

        def _validate_legs_self(self, legs: PauliArray) -> Literal[True]:
            """Validates the value of the :attr:`legs` property."""
            Gadget._validate_legs(legs)
            if len(legs) != self.num_qubits:
                raise ValueError("Number of legs does not match number of qubits.")
            return True

        def _validate_same_num_qubits(self, gadget: Gadget) -> Literal[True]:
            validate(gadget, Gadget)
            if self.num_qubits != gadget.num_qubits:
                raise ValueError("Mismatch in number of qubits between gadgets.")
            return True


class Layer:
    """A layer of Pauli gadgets with compatible legs."""

    @staticmethod
    def _subset_to_indicator(qubits: Iterable[int]) -> int:
        """Converts a collection of non-negative integers to the subset indicator."""
        ind = 0
        for i in qubits:
            ind |= 1 << i
        return ind

    @staticmethod
    def _selected_legs_to_subset(legs: PauliArray) -> int:
        """Convert legs to a subset index."""
        return Layer._subset_to_indicator(i for i, leg in enumerate(legs) if leg != 0)

    @staticmethod
    def _select_leg_subset(subset: int, legs: PauliArray) -> PauliArray:
        """Selects a subset of legs based on the given subset indicator."""
        return np.where(
            np.fromiter((subset & (1 << x) for x in range(len(legs))), dtype=np.bool_),
            legs,
            0,
        )

    @staticmethod
    def select_leg_subset(qubits: Iterable[int], legs: PauliArray) -> PauliArray:
        """Selects legs based on the given subset of qubits."""
        return Layer._select_leg_subset(Layer._subset_to_indicator(qubits), legs)

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

    def phase(self, legs: PauliArray) -> Phase:
        """
        Get the phase of the given legs in the layer, or :obj:`None` if the legs
        are incompatible with the layer.
        """
        if not self.is_compatible_with(legs):
            raise ValueError("Selected legs are incompatible with layer.")
        return self._phases.get(Layer._selected_legs_to_subset(legs), 0)

    def is_compatible_with(self, legs: PauliArray) -> bool:
        """Check if the legs are compatible with the current layer."""
        assert self._validate_legs_self(legs)
        self_legs = self._legs
        return bool(np.all((self_legs == legs) | (self_legs == 0) | (legs == 0)))

    def commutes_with(self, legs: PauliArray) -> bool:
        """Check if the legs commute with the current layer."""
        assert self._validate_legs_self(legs)
        self_legs = self._legs
        for subset in self._phases:
            subset_legs = Layer._select_leg_subset(subset, self_legs)
            ovlp = sum((subset_legs != legs) & (subset_legs != 0) & (subset_legs != 0))
            if ovlp % 2 != 0:
                return False
        return True

    @overload
    def add_gadget(self, gadget: Gadget, /) -> bool: ...

    @overload
    def add_gadget(self, legs: PauliArray, phase: Phase, /) -> bool: ...

    def add_gadget(
        self,
        gadget_or_legs: PauliArray | Gadget,
        phase: Phase | None = None
    ) -> bool:
        """Add a gadget to the layer."""
        if isinstance(gadget_or_legs, Gadget):
            legs = gadget_or_legs.legs
            phase = gadget_or_legs.phase
        else:
            legs = gadget_or_legs
            assert phase is not None
        if not self.is_compatible_with(legs):
            return False
        phases = self._phases
        subset = Layer._selected_legs_to_subset(legs)
        if subset in phases:
            if are_same_phase(curr_phase := phases[subset], -phase):
                del phases[subset]
                self._leg_count -= np.where(legs == 0, np.uint32(0), np.uint32(1))
                self._legs = np.where(self._leg_count == 0, 0, self._legs)
            else:
                phases[subset] = curr_phase+phase
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
            subset_legs = Layer._select_leg_subset(subset, legs)
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
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            return True

        def _validate_legs_self(self, legs: PauliArray) -> Literal[True]:
            """Validates the value of the :attr:`legs` property."""
            Gadget._validate_legs(legs)
            if len(legs) != self.num_qubits:
                raise ValueError("Number of legs does not match number of qubits.")
            return True
