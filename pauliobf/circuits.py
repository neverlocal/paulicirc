"""Circuits of Pauli gadgets."""

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
from collections.abc import Callable, Iterable, Iterator
from typing import (
    Any,
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
import autoray  # type: ignore[import-untyped]

from .spider_graphs import Matrix, SpiderGraph

from ._numpy import (
    RNG,
    BoolArray1D,
    Complex128Array1D,
    Complex128Array2D,
    UInt8Array1D,
    UInt8Array2D,
    normalise_phase,
    numba_jit,
)
from .gadgets import (
    PHASE_NBYTES,
    Gadget,
    GadgetData,
    Pauli,
    PauliArray,
    Phase,
    PhaseArray,
    decode_phases,
    encode_phases,
    gadget_data_len,
    get_phase,
    invert_phases,
    gadget_overlap,
    set_phase,
    get_gadget_legs,
)

if __debug__:
    from typing_validation import validate


CircuitData: TypeAlias = UInt8Array2D
"""Type alias for data encoding a circuit of Pauli gadgets."""

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
    if n % 4 != 0:
        # zeroes out the padding leg bits (up to 6 bits)
        mask = np.uint8(0b11111111 << 2 * (-n % 4) & 0b11111111)
        data[:, -PHASE_NBYTES - 1] &= mask
    # zeroes out the phase bytes
    data[:, -PHASE_NBYTES:] = 0
    return data


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

_GadgetDataTriple: TypeAlias = UInt8Array1D
"""
1D array containing the linearised data for three gadgets.

Data for the third gadget is set to zero, except for a commutation code
(cf. :obj:`CommutationCodeArray`) which has been written onto the last byte.
"""


@numba_jit
def _product_parity(p: GadgetData, q: GadgetData) -> int:
    p_legs = get_gadget_legs(p)
    q_legs = get_gadget_legs(q)
    s = 0
    for p_pauli, q_pauli in zip(p_legs, q_legs):
        if (p_pauli, q_pauli) in [(2, 1), (1, 3), (3, 2)]:
            s += 1
    return s % 2


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
    prod_parity = _product_parity(p, q)
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


def commute(circ: CircuitData, codes: CommutationCodeArray) -> CircuitData:
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


@final
class Circuit:
    """A quantum circuit, represented as a sequential composition of Pauli gadgets."""

    @classmethod
    def zero(cls, num_gadgets: int, num_qubits: int) -> Self:
        """
        Constructs a circuit with the given number of gadgets and qubits,
        where all gadgets have no legs and zero phase.
        """
        assert Circuit._validate_circ_shape(num_gadgets, num_qubits)
        data = _zero_circ(num_gadgets, num_qubits)
        return cls(data, num_qubits)

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
        return cls(data, num_qubits)

    @classmethod
    def random_inverse_pairs(
        cls, num_pairs: int, num_qubits: int, *, rng: int | RNG | None = None
    ) -> Self:
        """Constructs a circuit consisting of inverse pairs of random gadgets."""
        gadgets = Circuit.random(num_pairs, num_qubits, rng=rng)
        circ = Circuit.zero(2 * num_pairs, num_qubits)
        circ[::2] = gadgets
        gadgets.invert_phases()
        circ[1::2] = gadgets
        return circ

    @classmethod
    def from_gadgets(
        cls, gadgets: Iterable[Gadget], num_qubits: int | None = None
    ) -> Self:
        """Constructs a circuit from the given gadgets."""
        gadgets = list(gadgets)
        assert Circuit.__validate_gadgets(gadgets, num_qubits)
        if num_qubits is None:
            num_qubits = gadgets[0].num_qubits
        data = np.array([g._data for g in gadgets], dtype=np.uint8).reshape(
            len(gadgets), gadget_data_len(num_qubits)
        )
        return cls(data, num_qubits)

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
        return decode_phases(self._data[:, -PHASE_NBYTES:])

    @phases.setter
    def phases(self, value: PhaseArray) -> None:
        """Sets phases for the gadgets in the circuit."""
        assert self._validate_phases_value(value)
        self._data[:, -PHASE_NBYTES:] = encode_phases(value)

    def clone(self) -> Self:
        """Creates a copy of the gadget circuit."""
        return Circuit(self._data.copy(), self._num_qubits)

    def inverse(self) -> Self:
        """
        Returns the inverse of this graph, with both phases and gadget order inverted.
        """
        inverse = self[::-1].clone()
        inverse.invert_phases()
        return inverse

    def invert_phases(self) -> None:
        """Inverts phases inplace, keeping gadget order unchanged."""
        invert_phases(self._data[:, -PHASE_NBYTES:])

    def spider_graph(self) -> SpiderGraph:
        """
        Constructs and returns a spider graph modelling the tensor network for this
        circuit.
        The first :attr:`num_qubits` wires are the inputs of the circuit,
        the last :attr:`num_qubits` wires are the outputs of the circuit.
        """
        # 1. Extract numpy-like functions for chosen autoray backend.
        array: Callable[..., Matrix]
        matmul: Callable[[Matrix, Matrix], Matrix]
        array = autoray.numpy.array
        sqrt = autoray.numpy.sqrt
        exp = autoray.numpy.exp
        matmul = autoray.numpy.matmul
        # 2. Define cached matrices to be used when constructing the circuit.
        I: Matrix = array([[1 + 0j, 0], [0, 1 + 0j]])  # noqa: E741
        H: Matrix = array([[1 + 0j, 1 + 0j], [1 + 0j, -1 + 0j]]) / sqrt(2)
        S: Matrix = array([[1 + 0j, 0], [0, 1j]])
        S_dag: Matrix = array([[1 + 0j, 0], [0, -1j]])
        basis_change_start = [I, H, I, S_dag]
        basis_change_end = [I, H, I, S]
        basis_change_middle: dict[tuple[Pauli, Pauli], Matrix] = {
            (leg, prev_leg): basis_change_start[leg] @ basis_change_end[prev_leg]  # type: ignore
            for leg in range(4)
            for prev_leg in range(4)
        }

        def rot_z(phase: Phase) -> Matrix:
            return array([[exp(-1j * phase / 2), 0], [0, exp(1j * phase / 2)]])

        def rot_zh(phase: Phase) -> Matrix:
            return matmul(rot_z(phase), H)

        def rot_z_curr_prev(phase: Phase, curr: Pauli, prev: Pauli) -> Matrix:
            return matmul(rot_z(phase), basis_change_middle[(curr, prev)])

        # 3. Create spider graph with sufficient initial capacity.
        n, m = self.num_qubits, self.num_gadgets
        g = SpiderGraph(
            edge_capacity=(n + m * (2 * n + 1)), spider_capacity=(2 * n + m * (n + 2))
        )
        # 4. Assemble the spider graph.
        # The spiders currently on top of the circuit.
        # Initialised to be the circuit inputs (exactly num_qubits spiders).
        spiders = list(g.add_spiders((2,) * n))
        # The basis change to be applied to the spiders on top of the circuit.
        # The basis change for each spider is only applied when it is buried by
        # the next spider (worst case it happens at the end, with an output spider).
        prev_legs: PauliArray = np.zeros(n, dtype=np.uint8)
        for gadget_idx, gadget in enumerate(self.iter_gadgets(fast=True)):
            phase = gadget.phase
            if phase == 0:
                # Zero phase, skip the gadget.
                continue
            legs: PauliArray = gadget.legs
            num_legs = np.sum(legs != 0)
            if num_legs == 0:
                # Zero legs, skip the gadget.
                continue
            if num_legs == 1:
                # Add new leg spider.
                q = int(legs.argmax())
                h = g.add_spider(2)
                # Connect pre leg spider to new leg spider:
                # (prev leg spider)--|prev end|--|new start|--|z rot|--(new leg spider)
                g.add_edge(rot_z_curr_prev(phase, legs[q], prev_legs[q]), spiders[q], h)
                # Update spiders and prev legs.
                spiders[q] = h
                prev_legs[q] = legs[q]
                continue
            # Boolean flags indicating whether a new spider is created at each qubit.
            is_new_spider: BoolArray1D = (legs != prev_legs) & (legs != 0)
            num_new_spiders = np.sum(is_new_spider)
            # Add new leg spiders, new hub spider and new head spider.
            _spiders = list(
                map(
                    int,
                    np.where(
                        is_new_spider,
                        np.cumsum(is_new_spider, dtype=np.uint64) + g.num_spiders - 1,
                        spiders,
                    ),
                )
            )
            _new_spiders = g.add_spiders((2,) * (num_new_spiders + 2))
            hub_spider, head_spider = _new_spiders[-2:]
            # Connect prev leg spiders to new leg spiders:
            # (prev leg spider)--|prev end|--|new start|--(new leg spider)
            g.add_edges(
                (basis_change_middle[leg, prev_leg], t, h)
                for t, h, prev_leg, leg in zip(spiders, _spiders, prev_legs, legs)
                if leg != prev_leg and leg != 0  # only where new spider created
            )
            # Connect new leg spiders to new hub spider:
            # (new leg spider)--|H|--(new hub spider)
            g.add_edges(
                (basis_change_middle[1, leg], t, hub_spider)
                for t, leg in zip(_spiders, legs)
                if leg != 0
            )
            # Connect new hub spider to new head spider:
            # (new hub spider)--|H|--|z rot|--(new head spider)
            g.add_edge(rot_zh(phase), hub_spider, head_spider)
            # Update spiders and prev legs.
            spiders = _spiders
            prev_legs = np.where(is_new_spider, legs, prev_legs)
        # Add output spiders.
        output_spiders = g.add_spiders((2,) * n)
        # Connect leg spiders to output spiders:
        # (prev leg spider)--|prev end|--(output spider)
        g.add_edges(
            (basis_change_end[prev_leg], t, h)
            for t, h, prev_leg in zip(spiders, output_spiders, prev_legs)
        )
        # 5. Trim spider graph capacity and return.
        g.trim_capacity()
        return g

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
        if len(self) == 0:
            return self.clone()
        return Circuit(commute(self._data, codes), self._num_qubits)

    def random_commute(
        self, *, non_zero: bool = False, rng: int | RNG | None = None
    ) -> Self:
        """
        Commutes adjacent gadget pairs in the circuit according to randomly sampled
        commutation codes.
        """
        if len(self) == 0:
            return self.clone()
        codes = self.random_commutation_codes(non_zero=non_zero, rng=rng)
        return self.commute(codes)

    def unitary(
        self,
        *,
        _normalise_phase: bool = True,
        _use_cupy: bool = False,  # currently in alpha
    ) -> Complex128Array2D:
        """Returns the unitary matrix associated to this Pauli gadget circuit."""
        res: Complex128Array2D = np.eye(2**self.num_qubits, dtype=np.complex128)
        if _use_cupy:
            import cupy as cp  # type: ignore[import-untyped]

            res = cp.asarray(res)
        for gadget in self:
            gadget_u = gadget.unitary(_normalise_phase=False)
            if _use_cupy:
                gadget_u = cp.asarray(res)
            res = gadget_u @ res
        if _use_cupy:
            res = cp.asnumpy(res).astype(np.complex128)
        if _normalise_phase:
            normalise_phase(res)
        return res

    def statevec(
        self,
        input: Complex128Array1D,
        _normalise_phase: bool = True,
        _use_cupy: bool = False,  # currently in alpha
    ) -> Complex128Array1D:
        """
        Computes the statevector resulting from the application of this gadget circuit
        to the given input statevector.
        """
        assert validate(input, Complex128Array1D)
        res = input
        if _use_cupy:
            import cupy as cp

            res = cp.asarray(res)
        for gadget in self:
            gadget_u = gadget.unitary(_normalise_phase=False)
            if _use_cupy:
                gadget_u = cp.asarray(res)
            res = gadget_u @ res
        if _normalise_phase:
            normalise_phase(res)
        return res

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
    def __setitem__(self, idx: slice | list[SupportsIndex], value: Circuit) -> None: ...
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return (
            self.num_qubits == other.num_qubits
            and self.num_gadgets == other.num_gadgets
            and all(g == h for g, h in zip(self, other, strict=True))
        )

    def __repr__(self) -> str:
        m, n = self.num_gadgets, self.num_qubits
        return f"<Circuit: {m} gadgets, {n} qubits>"

    if __debug__:

        @staticmethod
        def _validate_circ_shape(num_gadgets: int, num_qubits: int) -> Literal[True]:
            """Validates the shape of a circuit."""
            validate(num_gadgets, SupportsIndex)
            validate(num_qubits, SupportsIndex)
            num_gadgets = int(num_gadgets)
            num_qubits = int(num_qubits)
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
                validate(num_qubits, SupportsIndex)
                num_qubits = int(num_qubits)
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
