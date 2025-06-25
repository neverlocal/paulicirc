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
from collections.abc import Iterable, Iterator
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
import numpy as np


from ._numpy import (
    RNG,
    Complex128Array1D,
    Complex128Array2D,
    UInt8Array1D,
    UInt8Array2D,
    normalise_phase,
)
from .gadgets import (
    PHASE_NBYTES,
    Gadget,
    PhaseArray,
    decode_phases,
    encode_phases,
    gadget_data_len,
    invert_phases,
    _aux_commute_pair,
)

if __debug__:
    from typing_validation import validate

try:
    import qiskit  # type: ignore[import-untyped]
    from qiskit import QuantumCircuit as QiskitQuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate as QiskitPauliEvolutionGate  # type: ignore[import-untyped]
    from qiskit.quantum_info import Pauli as QiskitPauli  # type: ignore[import-untyped]

except ModuleNotFoundError:
    pass


CircuitData: TypeAlias = UInt8Array2D
"""Type alias for data encoding a circuit of Pauli gadgets."""

CommutationCodeArray: TypeAlias = UInt8Array1D
"""
A 1D array of commutation codes, used by :meth:`Circuit.commute`.

See :class:`Gadget.commute_with` for a description of the commutation procedure
and associated commutation code conventions.
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


def commute(circ: CircuitData, codes: CommutationCodeArray) -> CircuitData:
    """
    Commutes subsequent gadget pairs in the circuit according to the given codes.
    Expects the number of codes to be ``m//2``, where ``m`` is the number of gadgets.

    See :class:`Gadget.commute_with` for a description of the commutation procedure
    and associated commutation code conventions.
    """
    m, _n = circ.shape
    _m = m + m // 2 + 2 * (m % 2)
    exp_circ = np.zeros((_m, _n), dtype=np.uint8)
    exp_circ[::3] = circ[::2]
    exp_circ[1 : _m - 2 * (m % 2) : 3] = circ[1::2]
    exp_circ[2 : _m - (m % 2) : 3, -1] = codes % 8
    reshaped_exp_circ = exp_circ.reshape(_m // 3, 3 * _n)
    np.apply_along_axis(_aux_commute_pair, 1, reshaped_exp_circ)
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

    def random_commutation_codes(
        self, *, non_zero: bool = False, rng: int | RNG | None = None
    ) -> CommutationCodeArray:
        """
        Returns an array of randomly sampled commutation codes for this circuit.

        If ``non_zero`` is set to :obj:`True`, commutation codes are all non-zero,
        forcing commutation for all pairs.

        See :class:`Gadget.commute_with` for a description of the commutation procedure
        and associated commutation code conventions.
        """
        if not isinstance(rng, RNG):
            rng = np.random.default_rng(rng)
        return rng.integers(int(non_zero), 8, self.num_gadgets // 2, dtype=np.uint8)

    def commute(self, codes: Sequence[int] | CommutationCodeArray) -> Self:
        """
        Commutes adjacent gadget pairs in the circuit according to the given commutation
        codes.

        See :class:`Gadget.commute_with` for a description of the commutation procedure
        and associated commutation code conventions.
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

        See :class:`Gadget.commute_with` for a description of the commutation procedure
        and associated commutation code conventions.
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
        self._data[idx, :] = value._data  # type: ignore[index]

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

    if "qiskit" in globals():

        def to_qiskit(self) -> QiskitQuantumCircuit:
            qiskit_circ = QiskitQuantumCircuit(num_qubits := self.num_qubits)
            for g in self:
                gate = QiskitPauliEvolutionGate(
                    QiskitPauli(g.leg_paulistr.replace("_", "I")), g.phase
                )
                qiskit_circ.append(gate, range(num_qubits))
            return qiskit_circ

    else:  # mock qiskit-specific methods

        def __to_qiskit(self) -> Any:
            raise ModuleNotFoundError("The 'qiskit' package is not installed.")

        __to_qiskit.__name__ = "to_qiskit"
        __to_qiskit.__qualname__ = "Circuit.to_qiskit"
        to_qiskit = __to_qiskit
        del __to_qiskit

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
