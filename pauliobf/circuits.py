"""Circuits of Pauli gadgets."""

from __future__ import annotations
from collections.abc import Iterable
from fractions import Fraction
from typing import (
    Final,
    Literal,
    Self,
    Sequence,
    SupportsIndex,
    TypeAlias,
    cast,
    final,
    overload,
)
import numpy as np

from ._numpy import (
    RNG,
    UInt16Array1D,
    UInt8Array1D,
    UInt8Array2D,
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

PAULI_CHARS: Final[Sequence[PauliChar]] = ("_", "X", "Z", "Y")
"""
Single-character representations of Paulis,
in order compatible with the chosen encoding (cf. :obj:`Pauli`).
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
r"""
Type alias for a phase, represented as the integer :math:`k` such that the phase
angle is equal to :math:`k\pi/32768` (see :obj:`PHASE_DENOM`).
"""

PhaseArray: TypeAlias = UInt16Array1D
"""Type alias for a 1D array of phases, as a 1D UInt16 array."""


assert PHASE_NBYTES == PhaseDataArray.__args__[0].__args__[1].__args__[0]  # type: ignore
assert PhaseArray.__args__[1].__args__[0].__name__ == f"uint{8*PHASE_NBYTES}"  # type: ignore


def _circ_ncols(m: int, n: int) -> int:
    """
    Number of columns for a circuit with ``m`` gadgets on ``n`` qubits.

    Presumes that the number ``n`` of qubits is divisible by 4.
    """
    legs_nbytes = -(-n // 4)
    return legs_nbytes + PHASE_NBYTES


def _shape(circ: GadgetCircData) -> tuple[int, int]:
    """
    Given gadget circuit data, returns the pair ``(m,n)`` of
    the number ``m`` of gadgets and the number ``n`` of qubits.
    """
    m, _n = circ.shape
    n = (_n - PHASE_NBYTES) * 4
    return m, n


def _get_phases(circ: GadgetCircData) -> PhaseDataArray:
    """Returns the array of phases for the gadgets in a given circuit."""
    return circ[:, -PHASE_NBYTES:]


def _set_phases(circ: GadgetCircData, phases: PhaseDataArray) -> None:
    """Sets the array of phases for the gadgets in the given circuit."""
    circ[:, -PHASE_NBYTES:] = phases


def _zero_circ(m: int, n: int) -> GadgetCircData:
    """
    Returns a circuit with ``m`` gadgets on ``n`` qubits,
    where all gadgets have no legs and zero phase.

    Presumes that the number ``n`` of qubits is divisible by 4.
    """
    return np.zeros((m, _circ_ncols(m, n)), dtype=np.uint8)


def _rand_circ(m: int, n: int, *, rng: RNG) -> GadgetCircData:
    """
    Returns a uniformly random circuit with ``m`` gadgets on ``n`` qubits,
    where all gadgets have no legs and zero phase.

    Presumes that the number ``n`` of qubits is divisible by 4.
    """
    data = rng.integers(0, 256, (m, _circ_ncols(m, n)), dtype=np.uint8)
    mask = 0b11111111 << 2 * (-n % 4) & 0b11111111
    data[:, -PHASE_NBYTES - 1] &= mask
    return data


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
def _phase2float(phase: Phase) -> float:
    """Converts a phase (as an int) to radians (as a float)."""
    return 2 * np.pi * phase / PHASE_DENOM


@numba_jit
def _float2phase(phase_f: float) -> Phase:
    """Converts radians (as a float) to a phase (as an int)."""
    return int(np.round(phase_f * PHASE_DENOM * 0.5 / np.pi)) % PHASE_DENOM


assert (
    PHASE_NBYTES == 2
), "Functions below are implemented under the assumption of 16-bit phases."


@numba_jit
def _get_phase(g: GadgetData) -> Phase:
    """Extracts phase data from the given gadget data."""
    return int(g[-2]) * 256 + int(g[-1])


@numba_jit
def _set_phase(g: GadgetData, phase: Phase) -> None:
    """Sets phase data in the given gadget data."""
    g[-2], g[-1] = divmod(phase, 256)


@final
class Gadget:
    """A Pauli gadget."""

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
            num_qubits = (data.shape[0]-PHASE_NBYTES) * 4
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
        return _get_gadget_legs(self._data)

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
        return _get_phase(self._data)

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
            _set_phase(self._data, value)
            return
        if isinstance(value, float):
            _set_phase(self._data, _float2phase(value))
            return
        validate(value, Fraction)
        num, den = value.numerator, value.denominator
        if (PHASE_DENOM//2)%den == 0:
            _set_phase(self._data, num*PHASE_DENOM//2//den)
            return
        _set_phase(self._data, _float2phase(num/den))

    @property
    def phase_float(self) -> float:
        r"""
        Approximate representation of the gadget phase,
        as a floating point number :math:`0 \leq x < \pi/32768`.
        """
        return _phase2float(self.phase)

    @property
    def phase_frac(self) -> Fraction:
        r"""
        Exact representation of the gadget phase as a fraction of :math:`\pi`.
        """
        return Fraction(self.phase, PHASE_DENOM//2)

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


@final
class GadgetCircuit:
    """A circuit of Pauli gadgets."""

    @classmethod
    def zero(cls, num_gadgets: int, num_qubits: int) -> Self:
        """
        Constructs a circuit with the given number of gadgets and qubits,
        where all gadgets have no legs and zero phase.
        """
        assert GadgetCircuit._validate_circ_shape(num_gadgets, num_qubits)
        data = _zero_circ(num_gadgets, num_qubits)
        return cls(data)

    @classmethod
    def random(
        cls, num_gadgets: int, num_qubits: int, *, rng: int | RNG | None = None
    ) -> Self:
        """
        Constructs a circuit with the given number of gadgets and qubits,
        where all gadgets have random legs and random phase.
        """
        assert GadgetCircuit._validate_circ_shape(num_gadgets, num_qubits)
        if not isinstance(rng, RNG):
            rng = np.random.default_rng(rng)
        data = _rand_circ(num_gadgets, num_qubits, rng=rng)
        return cls(data)

    _data: GadgetCircData
    _num_qubits: int

    def __new__(cls, data: GadgetCircData, num_qubits: int | None = None) -> Self:
        """
        Constructs a gadget circuit from the given data.

        :meta public:
        """
        assert GadgetCircuit._validate_new_args(data, num_qubits)
        if num_qubits is None:
            num_qubits = (data.shape[1]-PHASE_NBYTES) * 4
        self = super().__new__(cls)
        self._data = data
        self._num_qubits = num_qubits
        return self

    @property
    def num_gadgets(self) -> int:
        """Number of gadgets in the circuit."""
        return len(self._data)

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._num_qubits

    def iter_gadgets(self, *, fast: bool = False) -> Iterable[Gadget]:
        """
        Iterates over the gadgets in the circuit.

        If ``fast`` is set to ``True``, the gadgets yielded are ephemeral:
        they should not be stored, as the same object will be reused in each iteration.
        """
        if len(self._data) == 0:
            return
        if not fast:
            yield from iter(self)
            return
        g = Gadget(self._data[0], self._num_qubits, _ephemeral=True)
        for row in self._data:
            g._data = row
            yield g

    def __iter__(self) -> Iterable[Gadget]:
        """
        Iterates over the gadgets in the circuit.

        :meta public:
        """
        for row in self._data:
            yield Gadget(row, self._num_qubits)

    @overload
    def __getitem__(self, idx: SupportsIndex) -> Gadget: ...
    @overload
    def __getitem__(self, idx: slice | list[SupportsIndex]) -> GadgetCircuit: ...
    def __getitem__(
        self, idx: SupportsIndex | slice | list[SupportsIndex]
    ) -> Gadget | GadgetCircuit:
        """
        Accesses the gadget at a given index, or selects/slices a sub-circuit.

        :meta public:
        """
        if isinstance(idx, SupportsIndex):
            return Gadget(self._data[int(idx)], self._num_qubits)
        assert validate(idx, slice | list[SupportsIndex])
        return GadgetCircuit(self._data[idx, :], self._num_qubits)  # type: ignore[index]

    @overload
    def __setitem__(self, idx: SupportsIndex, value: Gadget) -> None: ...
    @overload
    def __setitem__(
        self, idx: slice | list[SupportsIndex], value: GadgetCircuit
    ) -> None: ...
    def __setitem__(
        self,
        idx: SupportsIndex | slice | list[SupportsIndex],
        value: Gadget | GadgetCircuit,
    ) -> None:
        """
        Writes a gadget at the given index of this circuit,
        or writes a sub-circuit onto the given selection/slice of this circuit.

        :meta public:
        """
        assert self._validate_setitem_args(idx, value)
        self._data[idx] = value._data

    def __len__(self) -> int:
        """
        Number of gadgets in the circuit.

        :meta public:
        """
        return len(self._data)

    def __repr__(self) -> str:
        m, n = self.num_gadgets, self.num_qubits
        return f"<GadgetCircuit: {m} gadgets, {n} qubits>"

    if __debug__:

        @staticmethod
        def _validate_circ_shape(num_gadgets: int, num_qubits: int) -> Literal[True]:
            """Validates the shape of a circuit."""
            validate(num_gadgets, int)
            validate(num_qubits, int)
            if num_gadgets < 0:
                raise ValueError("Number of gadgets must be non-negative.")
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            return True

        @staticmethod
        def _validate_new_args(
            data: GadgetCircData, num_qubits: int | None
        ) -> Literal[True]:
            """Validates the arguments of the :meth:`__new__` method."""
            validate(data, GadgetCircData)
            if num_qubits is not None:
                validate(num_qubits, int)
                if num_qubits < 0:
                    raise ValueError("Number of qubits must be non-negative.")
                if num_qubits > data.shape[1] * 4:
                    raise ValueError("Number of qubits exceeds circuit width.")
            return True

        def _validate_setitem_args(
            self,
            idx: SupportsIndex | slice | list[SupportsIndex],
            value: Gadget | GadgetCircuit,
        ) -> Literal[True]:
            """Validates the arguments to the :meth:`__setitem__` method."""
            if isinstance(idx, SupportsIndex):
                validate(value, Gadget)
            else:
                validate(value, GadgetCircuit)
                m_lhs = len(self._data[idx]) # type: ignore[index]
                m_rhs = cast(GadgetCircuit, value).num_gadgets
                if m_lhs != m_rhs:
                    raise ValueError(
                        "Mismatch in number of gadgets while writing sub-circuit:"
                        f"selection has {m_lhs} gadgets, rhs has {m_rhs}"
                    )
            if self.num_qubits != value.num_qubits:
                raise ValueError(
                    "Mismatch in number of qubits while writing circuit gadgets:"
                    f" lhs has {self.num_qubits} qubits, rhs has {value.num_qubits}."
                )
            return True
