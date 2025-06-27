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
from collections.abc import Iterable, Iterator
from fractions import Fraction
from math import lcm
import re
from typing import (
    Any,
    Final,
    Literal,
    Self,
    Sequence,
    SupportsIndex,
    TypeAlias,
    final,
    overload,
)
import euler
import numpy as np
from scipy.linalg import expm  # type: ignore[import-untyped]

from .utils.numpy import (
    RNG,
    Complex128Array1D,
    Complex128Array2D,
    ComplexArray1D,
    FloatArray1D,
    UInt8Array1D,
    UInt8Array2D,
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

PauliArray2D: TypeAlias = UInt8Array2D
"""
Type alias for a 2D array of Paulis, as 1D UInt8 array with entries in ``range(4)``.
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

PHASE_NBYTES: Final[int] = (
    int(re.match(r"float([0-9]+)", PHASE_DTYPE.__name__)[1]) // 8  # type: ignore
)
"""Number of bytes used for phase representation."""

assert PHASE_NBYTES >= 2, "Code presumes at least 16-bit precision."

# tuple[int, ...] used for all shapes to fix regression in Numpy 2.2.6 typing.
# Seems to be fine in Numpy 2.3, but that's currently not supported by Numba.

PhaseDataArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
"""
Type alias for a 1D array of encoded phase data,
as a 2D UInt8 NumPy array of shape ``(n, PHASE_NBYTES)``.
"""

PhaseArray: TypeAlias = FloatArray1D
"""Type alias for a 1D array of phases."""

QubitIdx: TypeAlias = int
"""Type alias for the index of a qubit."""

QubitIdxs: TypeAlias = Sequence[QubitIdx]
"""Type alias for a sequence of indexes of qubits."""


def broadcast_idxs(*idxs_seq: QubitIdx | QubitIdxs) -> tuple[QubitIdxs, ...]:
    """
    Broadcasts any number of sequences of qubit indexes to a common length,
    defined as the least common multiple of lengths for index sequences.
    Individual qubit indexes passed to this function are considered the same
    as singleton index sequences.
    If any sequence is empty, the returned sequences are all empty.
    """
    if not idxs_seq:
        return ()
    _idxs_seq: tuple[tuple[QubitIdx, ...], ...] = tuple(
        (idxs,) if isinstance(idxs, QubitIdx) else tuple(idxs) for idxs in idxs_seq
    )
    if any(not idxs for idxs in _idxs_seq):
        return ((),) * len(_idxs_seq)
    n = lcm(*map(len, _idxs_seq))
    return tuple(idxs * (n // len(idxs)) for idxs in _idxs_seq)


PhaseLike: TypeAlias = Phase | Fraction
r"""
Type alias for values which can be used to specify a phase:

- as a floating point value in :math:`[0, 2\pi)`, see :obj:`Phase`
- as a fraction of :math:`\pi`

"""


def gadget_data_len(num_qubits: int) -> int:
    """Returns the length of gadget data."""
    return -(-num_qubits // 4) + PHASE_NBYTES


def zero_gadget_data(num_qubits: int) -> GadgetData:
    """Returns blank data for a gadget with the given number of qubits."""
    return np.zeros(-(-num_qubits // 4) + PHASE_NBYTES, dtype=np.uint8)


# _LEG_BYTE_SHIFTS = np.arange(6, -1, -2, dtype=np.uint8)
# """Bit shifts ``[6, 4, 2, 0]`` used on a byte to extract leg information."""

# _LEG_BYTE_MASKS = 0b11 * np.ones(4, dtype=np.uint8)
# """Byte mask used on a byte to extract leg information."""


@numba_jit
def get_gadget_legs(g: GadgetData) -> PauliArray:
    """
    Extract an array of leg information from given gadget data.
    The returned array has values in ``range(4)``,
    where the encoding is explained in :obj:`GadgetData`.
    """
    leg_bytes = g[:-PHASE_NBYTES]
    n = len(leg_bytes)
    legs = np.zeros(4 * n, dtype=np.uint8)
    legs[::4] = (leg_bytes & 0b11_00_00_00) >> 6
    legs[1::4] = (leg_bytes & 0b00_11_00_00) >> 4
    legs[2::4] = (leg_bytes & 0b00_00_11_00) >> 2
    legs[3::4] = leg_bytes & 0b00_00_00_11
    return legs


def set_gadget_legs(g: GadgetData, legs: PauliArray) -> None:
    """
    Sets leg information in the given gadget data.
    The input array should have values in ``range(4)``,
    where the encoding is explained in :obj:`GadgetData`.
    """
    n = len(legs)
    leg_data = g[:-PHASE_NBYTES]
    leg_data[:] = 0
    leg_data[: -(-(n - 0) // 4)] |= legs[0::4] << 6  # type: ignore # ok in Numpy 2.3
    leg_data[: -(-(n - 1) // 4)] |= legs[1::4] << 4  # type: ignore # ok in Numpy 2.3
    leg_data[: -(-(n - 2) // 4)] |= legs[2::4] << 2  # type: ignore # ok in Numpy 2.3
    leg_data[: -(-(n - 3) // 4)] |= legs[3::4]


@numba_jit
def get_phase(g: GadgetData) -> Phase:
    """Extracts phase data from the given gadget data."""
    return float(g[-PHASE_NBYTES:].view(np.float64)[0])


@numba_jit
def set_phase(g: GadgetData, phase: Phase) -> None:
    """Sets phase data in the given gadget data."""
    g[-PHASE_NBYTES:] = np.array([phase % (2 * np.pi)], dtype=np.float64).view(np.uint8)


@numba_jit
def is_zero_phase(phase: Phase) -> bool:
    """Whether the given phase is deemed to be zero."""
    atol = 1e-8
    phase %= 2 * np.pi
    return bool(phase < atol or 2 * np.pi - phase < atol)


def are_same_phase(lhs: Phase, rhs: Phase) -> bool:
    """Whether the given phases are deemed to be the same."""
    from .utils.options import options

    lhs %= 2 * np.pi
    rhs %= 2 * np.pi
    return bool(np.isclose(lhs, rhs, options.rtol, options.atol))


def are_same_phases(lhs: PhaseArray, rhs: PhaseArray) -> bool:
    """Whether the given phase arrays are deemed to be the same."""
    from .utils.options import options

    lhs %= 2 * np.pi
    rhs %= 2 * np.pi
    return bool(np.isclose(lhs, rhs, options.rtol, options.atol).all())


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


_convert_0zx_yxz = numba_jit(euler.convert_xzx_yxz)
_convert_0zx_xyz = numba_jit(euler.convert_xzx_xyz)
_convert_0zx_xzy = numba_jit(euler.convert_xzx_xzy)
_convert_0zx_yxy = numba_jit(euler.convert_xzx_yxy)
_convert_0zx_yzy = numba_jit(euler.convert_xzx_yzy)
_convert_0zx_xyx = numba_jit(euler.convert_xzx_xyx)
_convert_0zx_zyz = numba_jit(euler.convert_xzx_zyz)

_convert_0xz_yzx = numba_jit(euler.convert_zxz_yzx)
_convert_0xz_zyx = numba_jit(euler.convert_zxz_zyx)
_convert_0xz_zxy = numba_jit(euler.convert_zxz_zxy)
_convert_0xz_yzy = numba_jit(euler.convert_zxz_yzy)
_convert_0xz_yxy = numba_jit(euler.convert_zxz_yxy)
_convert_0xz_zyz = numba_jit(euler.convert_zxz_zyz)
_convert_0xz_xyx = numba_jit(euler.convert_zxz_xyx)

CommutationCode: TypeAlias = int
"""
A number 0-7 describing how two Pauli gadgets are to be commuted past each other.
See :class:`Gadget.commute_past` for a description of the commutation procedure and
associated commutation code conventions.
"""

GadgetDataTriple: TypeAlias = UInt8Array1D
"""1D array containing the linearised data for three gadgets."""


@numba_jit
def pauli_product_parity(p: GadgetData, q: GadgetData) -> int:
    p_legs = get_gadget_legs(p)
    q_legs = get_gadget_legs(q)
    s = 0
    for p_pauli, q_pauli in zip(p_legs, q_legs):
        if (p_pauli, q_pauli) in [(2, 1), (1, 3), (3, 2)]:  # type: ignore[comparison-overlap]
            s += 1
    return s % 2


@numba_jit
def commute_gadget_pair(row: GadgetDataTriple) -> None:
    """
    Auxiliary function used by :func:`Gadget.commute_past` to commute a pair of gadgets.
    It operates on a single array, containing the linearised data for the two gadgets
    to be commuted, as well as auxiliary space for a third gadget.
    It expects data for the third gadget to be set to zero, except for the desired
    commutation code (cf. :obj:`CommutationCode`) which should be written on the
    last byte.
    """
    TOL = 1e-8
    n = len(row) // 3
    xi = row[-1]
    p: GadgetData = row[:n].copy()
    q: GadgetData = row[n : 2 * n].copy()
    a = get_phase(p)
    b = get_phase(q)
    if gadget_overlap(p, q) % 2 == 0:
        if xi != 0:
            row[2 * n :] = p
            row[:n] = 0
        return
    if xi == 0:
        return
    r = p ^ q  # phase bytes will be overwritten later
    prod_parity = pauli_product_parity(p, q)
    if xi < 3:
        if xi == 1:
            # 0zx -> zyz
            # 0qp -> qrq
            row[:n] = q
            row[n : 2 * n] = r
            row[2 * n :] = q
            if prod_parity == 0:
                _a, _b, _c = _convert_0zx_zyz(0, b, a, TOL)
            else:
                _a, _b, _c = _convert_0xz_xyx(0, b, a, TOL)
        else:  # xi == 2
            # 0zx -> yzy
            # 0qp -> rqr
            row[:n] = r
            row[n : 2 * n] = q
            row[2 * n :] = r
            if prod_parity == 0:
                _a, _b, _c = _convert_0zx_yzy(0, b, a, TOL)
            else:
                _a, _b, _c = _convert_0xz_yxy(0, b, a, TOL)
    elif xi < 5:
        if xi == 3:
            # 0zx -> xyx
            # 0qp -> prp
            row[:n] = p
            row[n : 2 * n] = r
            row[2 * n :] = p
            if prod_parity == 0:
                _a, _b, _c = _convert_0zx_xyx(0, b, a, TOL)
            else:
                _a, _b, _c = _convert_0xz_zyz(0, b, a, TOL)
        else:  # xi == 4
            # 0zx -> yxy
            # 0qp -> rpr
            row[:n] = r
            row[n : 2 * n] = p
            row[2 * n :] = r
            if prod_parity == 0:
                _a, _b, _c = _convert_0zx_yxy(0, b, a, TOL)
            else:
                _a, _b, _c = _convert_0xz_yzy(0, b, a, TOL)
    else:
        if xi == 5:
            # 0zx -> yxz
            # 0qp -> rpq
            row[:n] = q
            row[n : 2 * n] = p
            row[2 * n :] = r
            if prod_parity == 0:
                _a, _b, _c = _convert_0zx_yxz(0, b, a, TOL)
            else:
                _a, _b, _c = _convert_0xz_yzx(0, b, a, TOL)
        elif xi == 6:
            # 0zx -> xyz
            # 0qp -> prq
            row[:n] = q
            row[n : 2 * n] = r
            row[2 * n :] = p
            if prod_parity == 0:
                _a, _b, _c = _convert_0zx_xyz(0, b, a, TOL)
            else:
                _a, _b, _c = _convert_0xz_zyx(0, b, a, TOL)
        else:  # xi == 7
            # 0zx -> xzy
            # 0qp -> pqr
            row[:n] = r
            row[n : 2 * n] = q
            row[2 * n :] = p
            if prod_parity == 0:
                _a, _b, _c = _convert_0zx_xzy(0, b, a, TOL)
            else:
                _a, _b, _c = _convert_0xz_zxy(0, b, a, TOL)
    set_phase(row[:n], _c)
    set_phase(row[n : 2 * n], _b)
    set_phase(row[2 * n :], _a)


@final
class Gadget:
    """A Pauli gadget."""

    @staticmethod
    def phase2frac(phase: Phase) -> Fraction:
        r"""
        Converts a phase to a fraction of :math:`\pi`.
        The number of bits of precision is controlled by the value
        of :attr:`options.display_prec <paulicirc.utils.PauliCircOptions.display_prec>`:
        this is set to 8 by default, corresponding to multiples of :math:`\pi/256`.
        """
        from .utils.options import options

        K = 2**options.display_prec
        return Fraction(round(phase / np.pi * K) % (2 * K), K)

    @staticmethod
    def frac2phase(frac: Fraction) -> Phase:
        r"""Converts a fraction of :math:`\pi` to a phase (as a float)."""
        return (float(frac) * np.pi) % (2 * np.pi)

    @staticmethod
    def assemble_data(legs: PauliArray, phase: PhaseLike) -> GadgetData:
        """Assembles gadget data from the given legs and phase."""
        assert Gadget._validate_legs(legs)
        g = zero_gadget_data(len(legs))
        set_gadget_legs(g, legs)
        if isinstance(phase, Fraction):
            set_phase(g, Gadget.frac2phase(phase))
        else:
            set_phase(g, phase)
        return g

    @classmethod
    def zero(cls, num_qubits: int) -> Self:
        """Returns the gadget with no legs and zero phase."""
        data = zero_gadget_data(num_qubits)
        return cls(data, num_qubits)

    @classmethod
    def from_legs(cls, legs: PauliArray, phase: PhaseLike) -> Self:
        """Returns the gadget with given legs and phase."""
        assert Gadget._validate_legs(legs)
        num_qubits = len(legs)
        data = Gadget.assemble_data(legs, phase)
        return cls(data, num_qubits)

    @staticmethod
    def legs_from_paulistr(paulistr: str) -> PauliArray:
        """Returns the legs corresponding to the given paulistr."""
        assert Gadget._validate_paulistr(paulistr)
        return np.fromiter((PAULI_CHARS.index(p) for p in paulistr), dtype=np.uint8)

    @staticmethod
    def legs_from_sparse_paulistr(
        paulistr: str,
        qubits: QubitIdx | QubitIdxs,
        num_qubits: int,
    ) -> PauliArray:
        """Returns the legs corresponding to the given sparse paulistr."""
        if isinstance(qubits, QubitIdx):
            qubits = (qubits,)
        assert Gadget._validate_sparse_paulistr(paulistr, qubits, num_qubits)
        legs = np.zeros(num_qubits, dtype=np.uint8)
        for idx, p in zip(qubits, paulistr):
            legs[idx] = PAULI_CHARS.index(p)
        return legs

    @classmethod
    def from_paulistr(cls, paulistr: str, phase: PhaseLike) -> Self:
        """Returns the gadget with given legs (as paulistr) and phase."""
        legs = Gadget.legs_from_paulistr(paulistr)
        num_qubits = len(legs)
        data = Gadget.assemble_data(legs, phase)
        return cls(data, num_qubits)

    @classmethod
    def from_sparse_paulistr(
        cls,
        paulistr: str,
        qubits: QubitIdx | QubitIdxs,
        num_qubits: int,
        phase: PhaseLike,
    ) -> Self:
        """Returns the gadget with given legs (as a sparse paulistr) and phase."""
        legs = Gadget.legs_from_sparse_paulistr(paulistr, qubits, num_qubits)
        data = Gadget.assemble_data(legs, phase)
        return cls(data, num_qubits)

    @classmethod
    def random(
        cls,
        num_qubits: int,
        *,
        allow_zero: bool = True,
        allow_legless: bool = True,
        rng: int | RNG | None = None,
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
        phase: Phase = rng.uniform(0, 2 * np.pi)
        if not allow_zero:
            while is_zero_phase(phase):
                phase = rng.uniform(0, 2 * np.pi)
        data = Gadget.assemble_data(legs, phase)
        return cls(data, num_qubits)

    _data: GadgetData
    _num_qubits: int
    _ephemeral: bool

    __slots__ = ("__weakref__", "_data", "_num_qubits", "_ephemeral")

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
    def legs(self, value: PauliArray) -> None:
        """Sets the legs of the gadget."""
        assert validate(value, PauliArray | str | Sequence[Pauli])
        if isinstance(value, str):
            legs = Gadget.legs_from_paulistr(value)
        else:
            legs = np.asarray(value, dtype=np.uint8)
        assert self._validate_legs_self(legs)
        set_gadget_legs(self._data, legs)

    @property
    def leg_paulistr(self) -> str:
        """Paulistring representation of the gadget legs."""
        return "".join(PAULI_CHARS[int(p)] for p in self.legs)

    @property
    def phase(self) -> Phase:
        """Phase of the gadget, as an integer; see :obj:`Phase`."""
        return get_phase(self._data)

    @phase.setter
    def phase(self, value: Phase | Fraction) -> None:
        r"""Sets the phase of the gadget."""
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
        is_approx = Gadget.frac2phase(phase_frac) != self.phase
        prefix = "~" if is_approx else ""
        num, den = phase_frac.numerator, phase_frac.denominator
        if num == 0:
            return f"{prefix}0π"
        num_str = "" if num == 1 else str(num)
        if den == 1:
            return f"{prefix}{num_str}π"  # the only case should be 'π'
        return f"{prefix}{num_str}π/{str(den)}"

    @property
    def is_zero(self) -> bool:
        """Whether the gadget has zero phase."""
        return is_zero_phase(self.phase)

    @property
    def is_legless(self) -> bool:
        """Whether the gadget has no legs."""
        return bool(np.all(self.legs == 0))

    def inverse(self) -> Self:
        """
        Returns the inverse of this gadget, with phase negated.
        """
        g = self.clone()
        set_phase(g._data, -g.phase)
        return g

    def overlap(self, other: Gadget) -> int:
        """
        Returns the overlap between the legs of this gadgets and those of the given
        gadget, computed as the number of qubits where the legs of the two gadgets
        differ and are both not the identity Pauli (the value 0, as a :obj:`Pauli`).
        """
        assert self._validate_same_num_qubits(other)
        return gadget_overlap(self._data, other._data)

    def commutes_with(self, other: Gadget) -> bool:
        """
        Returns whether this gadget commutes with the given gadget,
        i.e. whether the overlap is even.
        """
        return not self.overlap(other) % 2

    def commute_past(
        self, other: Gadget, code: CommutationCode
    ) -> tuple[Gadget, Gadget, Gadget | None]:
        """
        Commutes this gadget past the given gadget, using the given
        :obj:`CommutationCode` to determine how the gadgets are to be commuted.

        If ``code=0``, the gadgets are not commuted:

        ..code-block:: python

            p.commute_past(q, 0) # -> (p, q, None)

        If the gadgets have even overlap, commutation always swaps them, regardless of
        the commutation code:

        ..code-block:: python

            # p.overlap(q) % 2 == 0
            p.commute_past(q, code) # -> (q, p, None)

        If the gadgets have odd overlap, commutation codes 1-7 correspond to the six
        possible ways to commute them, each resulting in a gadget triple:

        ..code-block:: python

            # p.overlap(q) % 2 != 0
            # code in range(1, 8)
            p.commute_past(q, code) # -> (r, s, t)

        The following mathematical procedure is used to compute the triple ``(r, s, t)``
        of gadgets obtained by commuting ``(p, q)``.
        Note that gadget ordering is read left-to-right, so that ``(p, q)`` means
        "p first, q second" and ``(r, s, t)`` means "r first, s second, r third";
        this is the opposite convention to matrix multiplication, which is read
        right-to-left instead.

        1. The gadgets ``(p, q)`` are simultaneously mapped, by means of a suitable
           Clifford circuit, to X and Z rotations on qubit 0.
           The rotations are either ``(x(0, p.phase), z(0, q.phase))``
           or ``(z(0, p.phase), x(0, q.phase))``, depending on whether there is an
           even or odd number of occurrences, respectively, of ZX, XY or YZ pairs in
           corresponding legs of ``p`` and ``q``.
        2. The X and Z rotations on qubit 0 are commuted past each other in one of six
           possible ways, described below and corresponding to commutation codes 1-7.
           This results in a sequence of three Pauli rotations on qubit 0, chosen
           between X, Y and Z rotations.
        3. The three Pauli rotations on qubit 0 are simultaneously mapped back to
           gadgets ``(r, s, t)`` by the inverse of the Clifford circuit from Step 1.

        The 6 possible commutations for Step 2 correspond to commutation codes 1-7 as
        follows, each commutation appearing in one of two flavours depending on whether
        Step 1 resulted in ``(rx, rz)`` or ``(rz, rx)``.

        - Code 1: ``(x, z) -> (z, y, z)`` and ``(z, x) -> (x, y, x)``
        - Code 2: ``(x, z) -> (y, z, y)`` and ``(z, x) -> (y, x, y)``
        - Code 3: ``(x, z) -> (x, y, x)`` and ``(z, x) -> (z, y, z)``
        - Code 4: ``(x, z) -> (y, x, y)`` and ``(z, x) -> (y, z, y)``
        - Code 5: ``(x, z) -> (z, x, y)`` and ``(z, x) -> (x, z, y)``
        - Code 6: ``(x, z) -> (z, y, x)`` and ``(z, x) -> (x, y, z)``
        - Code 7: ``(x, z) -> (y, z, x)`` and ``(z, x) -> (y, x, z)``

        """
        code %= 8
        if self.overlap(other) % 2 == 0:
            return (other, self, None)
        data = self._data
        num_qubits = self.num_qubits
        row: GadgetDataTriple = np.zeros(3 * (n := len(data)), dtype=np.uint8)
        row[:n] = data
        row[n : 2 * n] = other._data
        row[-1] = code
        commute_gadget_pair(row)
        return (
            Gadget(row[:n], num_qubits),
            Gadget(row[n : 2 * n], num_qubits),
            Gadget(row[2 * n :], num_qubits),
        )

    def unitary(self, *, normalize_phase: bool = True) -> Complex128Array2D:
        """Returns the unitary matrix associated to this Pauli gadget."""
        legs = self.legs
        if len(legs) == 0:
            return np.array([[np.exp(-0.5j * self.phase)]], dtype=np.complex128)
        kron_prod = PAULI_MATS[legs[0]]
        for leg in legs[1:]:
            kron_prod = np.kron(kron_prod, PAULI_MATS[leg])
        res: Complex128Array2D = expm(-0.5j * self.phase * kron_prod)
        if normalize_phase:
            normalise_phase(res)
        return res

    def statevec(
        self, input: ComplexArray1D | FloatArray1D, normalize_phase: bool = False
    ) -> Complex128Array1D:
        """
        Computes the statevector resulting from the application of this gadget
        to the given input statevector.
        """
        assert validate(input, ComplexArray1D | FloatArray1D)
        res = self.unitary(normalize_phase=False) @ input.astype(np.complex128)
        if normalize_phase:
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
        return f"<Gadget: {legs_str}, {self.phase_str}>"

    def __sizeof__(self) -> int:
        return (
            object.__sizeof__(self)
            + self._data.__sizeof__()
            + self._num_qubits.__sizeof__()
            + self._ephemeral.__sizeof__()
        )

    if __debug__:

        @staticmethod
        def _validate_paulistr(paulistr: str) -> Literal[True]:
            """Validates a given Paulistring."""
            validate(paulistr, str)
            if not all(p in PAULI_CHARS for p in paulistr):
                raise ValueError("Paulistring characters must be '_', 'X', 'Z' or 'Y'.")
            return True

        @staticmethod
        def _validate_sparse_paulistr(
            paulistr: str, qubits: QubitIdxs, num_qubits: int
        ) -> Literal[True]:
            """Validates a given Paulistring."""
            Gadget._validate_paulistr(paulistr)
            validate(qubits, QubitIdxs)
            validate(num_qubits, int)
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            if not all(0 <= q < num_qubits for q in qubits):
                raise ValueError("Invalid qubit indices.")
            if len(qubits) != len(set(qubits)):
                raise ValueError("Qubit indices cannot be repeated.")
            if len(qubits) != len(paulistr):
                raise ValueError(
                    "Paulistring length must match the number of specified qubit idxs."
                )
            return True

        @staticmethod
        def _validate_new_args(
            data: GadgetData, num_qubits: int | None
        ) -> Literal[True]:
            """Validates the arguments of the :meth:`__new__` method."""
            validate(data, GadgetData)
            if num_qubits is None:
                num_qubits = 4 * (len(data) - PHASE_NBYTES)
            else:
                validate(num_qubits, SupportsIndex)
                num_qubits = int(num_qubits)
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

    @classmethod
    def from_gadgets(
        cls, gadgets: Iterable[Gadget], num_qubits: int | None = None
    ) -> Self:
        """Constructs a layer from the given gadgets."""
        gadgets = list(gadgets)
        assert Layer.__validate_gadgets(gadgets, num_qubits)
        if num_qubits is None:
            num_qubits = gadgets[0].num_qubits
        self = cls(num_qubits)
        for gadget in gadgets:
            success = self.add_gadget(gadget)
            if not success:
                raise ValueError("Given gadgets do not form a single layer.")
        return self

    _phases: dict[int, Phase]
    _legs: PauliArray
    _leg_count: np.ndarray[tuple[int], np.dtype[np.uint32]]
    # FIXME: remove limit to 2**32 gadgets per layer

    __slots__ = ("__weakref__", "_phases", "_legs", "_leg_count")

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
        return "".join(PAULI_CHARS[int(p)] for p in self.legs)

    def phase(self, legs: PauliArray) -> Phase:
        """
        Get the phase of the given legs in the layer, or :obj:`None` if the legs
        are incompatible with the layer.
        """
        if not self.is_compatible_with(legs):
            raise ValueError("Selected legs are incompatible with layer.")
        return self._phases.get(Layer._selected_legs_to_subset(legs), 0)

    @overload
    def is_compatible_with(self, legs: PauliArray, /) -> bool: ...

    @overload
    def is_compatible_with(self, gadget: Gadget, /) -> bool: ...

    def is_compatible_with(self, legs: PauliArray | Gadget, /) -> bool:
        """Check if the legs are compatible with the current layer."""
        if isinstance(legs, Gadget):
            legs = legs.legs
        assert self._validate_legs_self(legs)
        self_legs = self._legs
        return bool(np.all((self_legs == legs) | (self_legs == 0) | (legs == 0)))

    @overload
    def commutes_with(self, legs: PauliArray, /) -> bool: ...

    @overload
    def commutes_with(self, gadget: Gadget, /) -> bool: ...

    def commutes_with(self, legs: PauliArray | Gadget, /) -> bool:
        """Check if the legs commute with the current layer."""
        if isinstance(legs, Gadget):
            legs = legs.legs
        assert self._validate_legs_self(legs)
        self_legs = self._legs
        for subset in self._phases:
            subset_legs = Layer._select_leg_subset(subset, self_legs)
            ovlp = sum((subset_legs != legs) & (legs != 0) & (subset_legs != 0))
            if ovlp % 2 != 0:
                return False
        return True

    @overload
    def add_gadget(self, gadget: Gadget, /) -> bool: ...

    @overload
    def add_gadget(self, legs: PauliArray, phase: Phase, /) -> bool: ...

    def add_gadget(
        self, gadget_or_legs: PauliArray | Gadget, phase: Phase | None = None
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
                phases[subset] = (curr_phase + phase) % (2 * np.pi)
            return True
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Layer):
            return NotImplemented
        print(self._phases)
        print(other._phases)
        return (
            self.num_qubits == other.num_qubits
            and np.array_equal(self._legs, other._legs)
            and self._phases == other._phases
        )

    def __repr__(self) -> str:
        legs_str = self.leg_paulistr
        if len(legs_str) > 16:
            legs_str = legs_str[:8] + "..." + legs_str[-8:]
        return f"<Layer: {legs_str}, {len(self)} gadgets>"

    def __sizeof__(self) -> int:
        return (
            object.__sizeof__(self)
            + self._phases.__sizeof__()
            + sum(
                key.__sizeof__() + value.__sizeof__()
                for key, value in self._phases.items()
            )
            + self._legs.__sizeof__()
            + self._leg_count.__sizeof__()
        )

    if __debug__:

        @staticmethod
        def _validate_new_args(num_qubits: int) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, SupportsIndex)
            num_qubits = int(num_qubits)
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            return True

        def _validate_legs_self(self, legs: PauliArray) -> Literal[True]:
            """Validates the value of the :attr:`legs` property."""
            Gadget._validate_legs(legs)
            if len(legs) != self.num_qubits:
                raise ValueError("Number of legs does not match number of qubits.")
            return True

        @staticmethod
        def __validate_gadgets(
            gadgets: Sequence[Gadget], num_qubits: int | None
        ) -> Literal[True]:
            validate(gadgets, Sequence[Gadget])
            if num_qubits is None:
                if not gadgets:
                    raise ValueError(
                        "At least one gadget must be supplied if num_qubits is omitted."
                    )
                num_qubits = gadgets[0].num_qubits
            for gadget in gadgets:
                if gadget.num_qubits != num_qubits:
                    raise ValueError("All gadgets must have the same number of qubits.")
            return True
