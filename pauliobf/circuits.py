"""Circuits of Pauli gadgets."""

from __future__ import annotations
from collections.abc import Iterable, Iterator
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
import euler
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

CircuitData: TypeAlias = UInt8Array2D
"""Type alias for data encoding a circuit of Pauli gadgets."""

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

CommutationCodeArray: TypeAlias = UInt8Array1D
"""
Commutation codes are integers in ``range(8)`` indicating how to commute gadets:
0 means no commutation, values 1-7 means commutation.

If the gadgets have even overlap, the commutation performed on codes 1-7 is
always the same, ``xz -> zx``.
If the gadgets have odd overlap, the commutations performed on codes 1-7 are as follows:

- 1 for ``xz -> zyz``
- 2 for ``xz -> yzy``
- 3 for ``xz -> xyx``
- 4 for ``xz -> yxy``
- 5 for ``xz -> yzx``
- 6 for ``xz -> zyx``
- 7 for ``xz -> zxy``

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

assert PHASE_NBYTES == PhaseDataArray.__args__[0].__args__[1].__args__[0]  # type: ignore
assert PhaseArray.__args__[1].__args__[0].__name__ == f"uint{8*PHASE_NBYTES}"  # type: ignore


def _zero_circ(m: int, n: int) -> CircuitData:
    """
    Returns a circuit with ``m`` gadgets on ``n`` qubits,
    where all gadgets have no legs and zero phase.

    Presumes that the number ``n`` of qubits is divisible by 4.
    """
    ncols = PHASE_NBYTES - (-n // 4)
    return np.zeros((m, ncols), dtype=np.uint8)


def _rand_circ(m: int, n: int, *, rng: RNG) -> CircuitData:
    """
    Returns a uniformly random circuit with ``m`` gadgets on ``n`` qubits,
    where all gadgets have no legs and zero phase.

    Presumes that the number ``n`` of qubits is divisible by 4.
    """
    ncols = PHASE_NBYTES - (-n // 4)
    data = rng.integers(0, 256, (m, ncols), dtype=np.uint8)
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


@numba_jit
def _overlap(p: GadgetData, q: GadgetData) -> int:
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


_convert_xz0_yzx = numba_jit(euler.convert_xzx_yzx)
_convert_xz0_zyx = numba_jit(euler.convert_xzx_zyx)
_convert_xz0_zxy = numba_jit(euler.convert_xzx_zxy)
_convert_xz0_yxy = numba_jit(euler.convert_xzx_yxy)
_convert_xz0_yzy = numba_jit(euler.convert_xzx_yzy)
_convert_xz0_xyx = numba_jit(euler.convert_xzx_xyx)
_convert_xz0_zyz = numba_jit(euler.convert_xzx_zyz)

_GadgetDataTriple: TypeAlias = UInt8Array1D
"""
1D array containing the linearised data for three gadgets.

Data for the third gadget is set to zero, except for a commutation code
(cf. :obj:`CommutationCodeArray`) which has been written onto the last byte.
"""


@numba_jit
def _aux_commute_pair(row: _GadgetDataTriple) -> None:
    """
    Auxiliary function used by :func:`_commute` to commute an adjacent pair of gadgets.
    Presumes that a third, zero gadget has been inserted after the two gadgets,
    and that the data for the tree gadgets was linearised; see :obj:`_GadgetTriple`.
    """
    TOL = 1e-8
    n = len(row) // 3
    xi = row[-1]
    p: GadgetData = row[:n]
    q: GadgetData = row[n : 2 * n]
    a = _phase2float(_get_phase(p))
    b = _phase2float(_get_phase(q))
    if _overlap(p, q) % 2 == 0:
        if xi != 0:
            row[2 * n :] = p
            row[:n] = 0
        return
    r = p ^ q  # phase bytes will be overwritten later
    if xi < 3:
        if xi == 1:
            # xz -> zyz
            row[:n] = q
            row[2 * n :] = q
            row[n : 2 * n] = r
            _a, _b, _c = _convert_xz0_zyz(a, b, 0, TOL)
        else:  # xi == 2
            # xz -> yzy
            row[:n] = r
            row[2 * n :] = r
            _a, _b, _c = _convert_xz0_yzy(a, b, 0, TOL)
    elif xi < 5:
        if xi == 3:
            # xz -> xyx
            row[2 * n :] = p
            row[n : 2 * n] = r
            _a, _b, _c = _convert_xz0_xyx(a, b, 0, TOL)
        else:  # xi == 4
            # xz -> yxy
            row[n : 2 * n] = p
            row[:n] = r
            row[2 * n :] = r
            _a, _b, _c = _convert_xz0_yxy(a, b, 0, TOL)
    else:
        if xi == 5:
            # xz -> yzx
            row[2 * n :] = p
            row[:n] = r
            _a, _b, _c = _convert_xz0_yzx(a, b, 0, TOL)
        elif xi == 6:
            # xz -> zyx
            row[2 * n :] = p
            row[:n] = q
            row[n : 2 * n] = r
            _a, _b, _c = _convert_xz0_zyx(a, b, 0, TOL)
        else:  # xi == 7
            # xz -> zxy
            q = q.copy()
            row[n : 2 * n] = p
            row[:n] = q
            row[2 * n :] = r
            _a, _b, _c = _convert_xz0_zxy(a, b, 0, TOL)
    _a_phase, _b_phase, _c_phase = _float2phase(_a), _float2phase(_b), _float2phase(_c)
    _set_phase(row[:n], _a_phase)
    _set_phase(row[n : 2 * n], _b_phase)
    _set_phase(row[2 * n :], _c_phase)


def _commute(circ: CircuitData, codes: CommutationCodeArray) -> CircuitData:
    """
    Commutes subsequent gadget pairs in the circuit according to the given codes;
    see :func:`_aux_commute_pair`.
    Expects the number of codes to be ``m//2``, where ``m`` is the number of gadgets.
    """
    m, _n = circ.shape
    _m = m + m // 2 + 2 * (m % 2)
    exp_circ = np.zeros((_m, _n), dtype=np.uint8)
    exp_circ[::3] = circ[::2]
    exp_circ[1 : _m - 2 * (m % 2) : 3] = circ[1::2]
    exp_circ[2 : _m - (m % 2) : 3, -1] = codes % 8
    reshaped_exp_circ = exp_circ.reshape(_m // 3, 3 * _n)
    np.apply_along_axis(_aux_commute_pair, 1, reshaped_exp_circ)  # type: ignore
    return exp_circ[~np.all(exp_circ == 0, axis=1)]  # type: ignore


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


@numba_jit
def _decode_phases(phases: PhaseDataArray) -> PhaseArray:
    """Decodes phase data from a gadget circuit into an array of phases."""
    return (phases[:, 0].astype(np.uint16) * 256 + phases[:, 1]).astype(np.uint16)


@numba_jit
def _encode_phases(phases: PhaseArray) -> PhaseDataArray:
    """Encodes an array of phases into phase data for a gadget circuit."""
    return np.vstack(np.divmod(phases, 256)).astype(np.uint8).T


@numba_jit
def _invert_phases(phases: PhaseDataArray) -> None:
    """Inplace phase inversion for the given phase data."""
    phases[:, 0], phases[:, 1] = np.divmod(
        PHASE_DENOM - (256 * phases[:, 0].astype(np.uint32) + phases[:, 1]), 256
    )


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
        if (PHASE_DENOM // 2) % den == 0:
            _set_phase(self._data, num * PHASE_DENOM // 2 // den)
            return
        _set_phase(self._data, _float2phase(num / den))

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
        return Fraction(self.phase, PHASE_DENOM // 2)

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
        return _overlap(self._data, other._data)

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


@final
class Circuit:
    """A circuit of Pauli gadgets."""

    @classmethod
    def zero(cls, num_gadgets: int, num_qubits: int) -> Self:
        """
        Constructs a circuit with the given number of gadgets and qubits,
        where all gadgets have no legs and zero phase.
        """
        assert Circuit._validate_circ_shape(num_gadgets, num_qubits)
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
        assert Circuit._validate_circ_shape(num_gadgets, num_qubits)
        if not isinstance(rng, RNG):
            rng = np.random.default_rng(rng)
        data = _rand_circ(num_gadgets, num_qubits, rng=rng)
        return cls(data)

    _data: CircuitData
    _num_qubits: int

    def __new__(cls, data: CircuitData, num_qubits: int | None = None) -> Self:
        """
        Constructs a gadget circuit from the given data.

        :meta public:
        """
        assert Circuit._validate_new_args(data, num_qubits)
        if num_qubits is None:
            num_qubits = (data.shape[1] - PHASE_NBYTES) * 4
        self = super().__new__(cls)
        self._data = data
        self._num_qubits = num_qubits
        return self

    def clone(self) -> Self:
        """Creates a copy of the gadget circuit."""
        return Circuit(self._data.copy(), self._num_qubits)

    @property
    def num_gadgets(self) -> int:
        """Number of gadgets in the circuit."""
        return len(self._data)

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._num_qubits

    @property
    def phases(self) -> PhaseArray:
        """Array of phases for the gadgets in the circuit."""
        return _decode_phases(self._data[:, -PHASE_NBYTES:])

    @phases.setter
    def phases(self, value: PhaseArray) -> None:
        """Sets phases for the gadgets in the circuit."""
        assert self._validate_phases_value(value)
        self._data[:, -PHASE_NBYTES:] = _encode_phases(value)

    def inverse(self) -> Self:
        inverse = self[::-1].clone()
        inverse.invert_phases()
        return inverse

    def invert_phases(self) -> None:
        """Inverts phases inplace, keeping gadget order unchanged."""
        _invert_phases(self._data[:, -PHASE_NBYTES:])

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

    def random_commutation_codes(
        self, *, non_zero: bool = False, rng: int | RNG | None = None
    ) -> CommutationCodeArray:
        """
        Returns an array of randomly sampled commutation codes for this circuit;
        see :obj:`CommutationCodeArray`.

        If ``non_zero`` is set to :obj:`True`, commutation codes are all non-zero,
        forcing commutation for all pairs.
        """
        if not isinstance(rng, RNG):
            rng = np.random.default_rng(rng)
        return rng.integers(int(non_zero), 8, self.num_gadgets // 2, dtype=np.uint8)

    def commute(self, codes: Sequence[int] | CommutationCodeArray) -> Self:
        """
        Commutes adjacent gadget pairs in the circuit according to the given commutation
        codes; see :obj:`CommutationCodeArray`.
        """
        codes = np.asarray(codes, dtype=np.uint8)
        assert self._validate_commutation_codes(codes)
        return Circuit(_commute(self._data, codes), self._num_qubits)

    def __iter__(self) -> Iterator[Gadget]:
        """
        Iterates over the gadgets in the circuit.

        :meta public:
        """
        for row in self._data:
            yield Gadget(row, self._num_qubits)

    @overload
    def __getitem__(self, idx: SupportsIndex) -> Gadget: ...
    @overload
    def __getitem__(self, idx: slice | list[SupportsIndex]) -> Circuit: ...
    def __getitem__(
        self, idx: SupportsIndex | slice | list[SupportsIndex]
    ) -> Gadget | Circuit:
        """
        Accesses the gadget at a given index, or selects/slices a sub-circuit.

        :meta public:
        """
        if isinstance(idx, SupportsIndex):
            return Gadget(self._data[int(idx)], self._num_qubits)
        assert validate(idx, slice | list[SupportsIndex])
        return Circuit(self._data[idx, :], self._num_qubits)  # type: ignore[index]

    @overload
    def __setitem__(self, idx: SupportsIndex, value: Gadget) -> None: ...
    @overload
    def __setitem__(
        self, idx: slice | list[SupportsIndex], value: Circuit
    ) -> None: ...
    def __setitem__(
        self,
        idx: SupportsIndex | slice | list[SupportsIndex],
        value: Gadget | Circuit,
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
            data: CircuitData, num_qubits: int | None
        ) -> Literal[True]:
            """Validates the arguments of the :meth:`__new__` method."""
            validate(data, CircuitData)
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
            value: Gadget | Circuit,
        ) -> Literal[True]:
            """Validates the arguments to the :meth:`__setitem__` method."""
            if isinstance(idx, SupportsIndex):
                validate(value, Gadget)
            else:
                validate(value, Circuit)
                m_lhs = len(self._data[idx])  # type: ignore[index]
                m_rhs = cast(Circuit, value).num_gadgets
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

        def _validate_phases_value(self, value: PhaseArray) -> Literal[True]:
            """Validates the value of the :attr:`phases` property."""
            validate(value, PhaseArray)
            if len(value) != self.num_gadgets:
                raise ValueError("Number of phases does not match number of gadgets.")
            return True

        def _validate_commutation_codes(
            self, codes: CommutationCodeArray
        ) -> Literal[True]:
            """Validates commutation codes passed to :meth:`commute`."""
            if len(codes) != self.num_gadgets // 2:
                raise ValueError(
                    f"Expected {self.num_gadgets//2} communication codes,"
                    f"found {len(codes)} instead."
                )
            if np.any(codes >= 8):
                raise ValueError("Communication codes must be in range(8).")
            return True
