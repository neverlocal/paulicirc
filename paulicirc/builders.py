"""Circuit builders."""

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
from collections.abc import Iterator, Sequence
from fractions import Fraction
from math import ceil
from typing import (
    Literal,
    Self,
    SupportsIndex,
    TypeAlias,
)

import numpy as np

from .utils.numpy import Complex128Array1D, Complex128Array2D, canonicalize_phase
from .gadgets import (
    Gadget,
    Phase,
    QubitIdx,
    QubitIdxs,
    set_gadget_leg_at,
    set_gadget_legs_at,
    set_phase,
)
from .circuits import Circuit, transversal_set_gadget_leg_at, transversal_set_phase

if __debug__:
    from typing_validation import validate

PhaseLike: TypeAlias = Phase | Fraction
r"""
Type alias for values which can be used to specify a phase:

- as a floating point value in :math:`[0, 2\pi)`, see :obj:`Phase`
- as a fraction of :math:`\pi`

"""


class CircuitBuilder:
    """A utility class to build circuits using common gate-based syntax."""

    _circuit: Circuit
    _num_gadgets: int
    _capacity_scaling: int | float

    __slots__ = ("__weakref__", "_circuit", "_num_gadgets", "_capacity_scaling")

    def __new__(
        cls,
        num_qubits: int,
        *,
        init_capacity: int | None = None,
        capacity_scaling: int | float = 2,
    ) -> Self:
        """
        Create an empty circuit builder with the given number of qubits.

        Optionally, initial capacity and/or capacity scaling can be specified
        for the underlying circuit. By default, the initial capacity is set to the
        number of qubits (with a minimum of 1), while the capacity scaling factor
        is set to 2 (doubling capacity whenever required).

        :meta public:
        """
        if init_capacity is None:
            init_capacity = max(1, num_qubits)
        assert CircuitBuilder.__validate_new_args(
            num_qubits, init_capacity, capacity_scaling
        )
        self = super().__new__(cls)
        self._circuit = Circuit.zero(init_capacity, num_qubits)
        self._num_gadgets = 0
        self._capacity_scaling = capacity_scaling
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits for the circuit."""
        return self._circuit.num_qubits

    @property
    def capacity(self) -> int:
        """The current capacity of the circuit builder."""
        return len(self._circuit)

    def circuit(self) -> Circuit:
        """Returns a circuit constructed from the gadgets currently in the builder."""
        return self._circuit[: self._num_gadgets].clone()

    def append(self, gadget: Gadget) -> None:
        """Appends a single gadget to the circuit being built."""
        if len(self) >= self.capacity:
            self._scale_up_capacity(1)
        self._circuit[len(self)] = gadget
        self._num_gadgets += 1

    def extend(self, gadgets: Sequence[Gadget] | Circuit) -> None:
        """
        Appends the given gadgets to the circuit being built.
        Gadgets are all validated prior to any modification,
        so either they are all appended to the circuit or none is.
        """
        if isinstance(gadgets, Circuit):
            new_circuit = gadgets
        else:
            new_circuit = Circuit.from_gadgets(gadgets, self.num_qubits)
        num_gadgets = len(self)
        num_new_gadgets = len(new_circuit)
        if num_gadgets + num_new_gadgets > self.capacity:
            self._scale_up_capacity(num_new_gadgets)
        self._circuit[num_gadgets : num_gadgets + num_new_gadgets] = new_circuit
        self._num_gadgets += num_new_gadgets

    def _scale_up_capacity(self, num_new_gadgets: int) -> None:
        capacity = len(self._circuit) * 1.0
        capacity_scaling = self._capacity_scaling
        target_capacity = len(self) + num_new_gadgets
        while capacity < target_capacity:
            capacity *= capacity_scaling
        self.set_capacity(int(ceil(capacity)))

    def set_capacity(self, new_capacity: int) -> None:
        """Sets the circuit capacity to the given value."""
        assert self.__validate_capacity(new_capacity)
        circuit = self._circuit
        capacity = len(circuit)
        if new_capacity == capacity:
            return
        ext_circuit = Circuit.zero(new_capacity, self.num_qubits)
        ext_circuit[:capacity] = circuit
        self._circuit = ext_circuit

    def trim_capacity(self) -> None:
        """Sets the circuit capacity to the minimum amount possible."""
        self.set_capacity(max(1, len(self)))

    def unitary(self, *, _normalise_phase: bool = True) -> Complex128Array2D:
        """Returns the unitary matrix associated to the circuit being built."""
        res = np.eye(2**self.num_qubits, dtype=np.complex128)
        for gadget in self:
            res = gadget.unitary(canonical_phase=False) @ res
        if _normalise_phase:
            canonicalize_phase(res)
        return res

    def statevec(
        self, input: Complex128Array1D, _normalise_phase: bool = False
    ) -> Complex128Array1D:
        """
        Computes the statevector resulting from the application of the circuit being
        built to the given input statevector.
        """
        assert validate(input, Complex128Array1D)
        res = input
        for gadget in self:
            res = gadget.unitary(canonical_phase=False) @ res
        if _normalise_phase:
            canonicalize_phase(res)
        return res

    def __iter__(self) -> Iterator[Gadget]:
        """Iterates over the gadgets currently in the circuit."""
        yield from self._circuit[: self._num_gadgets]

    def __len__(self) -> int:
        """The number of gadgets currently in the circuit."""
        return self._num_gadgets

    def __repr__(self) -> str:
        m, n = len(self), self.num_qubits
        return f"<CircuitBuilder: {m} gadgets, {n} qubits>"

    def __sizeof__(self) -> int:
        return (
            object.__sizeof__(self)
            + self._num_gadgets.__sizeof__()
            + self._capacity_scaling.__sizeof__()
            + self._circuit.__sizeof__()
        )

    def _gadget(self, ps: Sequence[int], qs: QubitIdxs, phase: PhaseLike) -> None:
        """Adds a gadget with given Paulistring and angle on given qubits."""
        assert CircuitBuilder.__validate_qubit_idxs(qs)
        if isinstance(phase, Fraction):
            phase = (float(phase) * np.pi) % (2 * np.pi)
        if len(self) == self.capacity:
            self._scale_up_capacity(1)
        gadget_data = self._circuit._data[self._num_gadgets]
        set_gadget_legs_at(gadget_data, np.array(ps, np.uint8), np.array(qs, np.uint64))
        set_phase(gadget_data, phase)
        self._num_gadgets += 1

    # TODO: add support for transversal gadget application with disjoint leg qubits

    def _rot(self, p: int, q: QubitIdx | QubitIdxs, phase: PhaseLike) -> None:
        assert CircuitBuilder.__validate_qubit_idxs(q)
        if isinstance(phase, Fraction):
            phase = (float(phase) * np.pi) % (2 * np.pi)
        if isinstance(q, QubitIdx):
            if len(self) == self.capacity:
                self._scale_up_capacity(1)
            gadget_data = self._circuit._data[self._num_gadgets]
            set_gadget_leg_at(gadget_data, p, q)
            set_phase(gadget_data, phase)
            self._num_gadgets += 1
        else:
            num_qs = len(q)
            if len(self) + num_qs > self.capacity:
                self._scale_up_capacity(num_qs)
            start = self._num_gadgets
            end = start + num_qs
            circ = self._circuit._data
            transversal_set_gadget_leg_at(circ, p, np.array(q, np.uint64), start, end)
            transversal_set_phase(circ, phase, start, end)
            self._num_gadgets += num_qs

    def rx(self, phase: PhaseLike, q: QubitIdx | QubitIdxs) -> None:
        """Adds an X rotation with given angle on the given qubit(s)."""
        self._rot(0b01, q, phase)

    def rz(self, phase: PhaseLike, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Z rotation with given angle on the given qubit(s)."""
        self._rot(0b10, q, phase)

    def ry(self, phase: PhaseLike, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Y rotation with given angle on the given qubit(s)."""
        self._rot(0b11, q, phase)

    def x(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a X gate on the given qubit."""
        self._rot(0b01, q, Fraction(1, 1))

    def z(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Z gate on the given qubit."""
        self._rot(0b10, q, Fraction(1, 1))

    def y(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Y gate on the given qubit."""
        self._rot(0b11, q, Fraction(1, 1))

    def sx(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a √X gate on the given qubit."""
        self._rot(0b01, q, Fraction(1, 2))

    def sxdg(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a √X† gate on the given qubit."""
        self._rot(0b01, q, Fraction(-1, 2))

    def s(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a S gate on the given qubit."""
        self._rot(0b10, q, Fraction(1, 2))

    def sdg(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a S† gate on the given qubit."""
        self._rot(0b10, q, Fraction(-1, 2))

    def t(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a T gate on the given qubit."""
        self._rot(0b10, q, Fraction(1, 4))

    def tdg(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a T† gate on the given qubit."""
        self._rot(0b10, q, Fraction(-1, 4))

    def h(self, q: QubitIdx | QubitIdxs, *, xzx: bool = False) -> None:
        """
        Adds a H gate on the given qubit.

        By default, this is decomposed as ``Z(pi/2)X(pi/2)Z(pi/2)``,
        but setting ``xzx=True`` decomposes it as ``X(pi/2)Z(pi/2)X(pi/2)`` instead.
        """
        b = 0b01 if xzx else 0b10
        self._rot(b, q, Fraction(1, 2))
        self._rot(3 - b, q, Fraction(1, 2))  # 0b10 if xzx else 0b01
        self._rot(b, q, Fraction(1, 2))

    def hdg(self, q: QubitIdx | QubitIdxs, *, xzx: bool = False) -> None:
        """
        Adds a H gate on the given qubit.

        By default, this is decomposed as ``Z(-pi/2)X(-pi/2)Z(-pi/2)``,
        but setting ``xzx=True`` decomposes it as ``X(-pi/2)Z(-pi/2)X(-pi/2)`` instead.
        """
        b = 0b01 if xzx else 0b10
        self._rot(b, q, Fraction(-1, 2))
        self._rot(3 - b, q, Fraction(-1, 2))  # 0b10 if xzx else 0b01
        self._rot(b, q, Fraction(-1, 2))

    def cz(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CZ gate to the given control and target qubits."""
        self._rot(0b10, c, Fraction(-1, 2))
        self._rot(0b10, t, Fraction(-1, 2))
        self._gadget([2, 2], [c, t], Fraction(1, 2))

    def cx(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CX gate to the given control and target qubits."""
        self._rot(0b10, t, Fraction(1, 2))
        self._rot(0b01, t, Fraction(1, 2))
        self._rot(0b10, c, Fraction(-1, 2))
        self._rot(0b10, t, Fraction(-1, 2))
        self._gadget([2, 2], [c, t], Fraction(1, 2))
        self._rot(0b01, t, Fraction(-1, 2))
        self._rot(0b10, t, Fraction(-1, 2))

    def cy(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CY gate to the given control and target qubits."""
        self._rot(0b01, t, Fraction(1, 2))
        self._rot(0b10, c, Fraction(-1, 2))
        self._rot(0b10, t, Fraction(-1, 2))
        self._gadget([2, 2], [c, t], Fraction(1, 2))
        self._rot(0b01, t, Fraction(-1, 2))

    def swap(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a SWAP gate to the given control and target qubits."""
        self.cx(c, t)
        self.cx(t, c)
        self.cx(c, t)

    def ccx(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCX gate to the given control and target qubits."""
        self._rot(0b10, t, Fraction(1, 2))
        self._rot(0b01, t, Fraction(1, 2))
        self.ccz(c0, c1, t)
        self._rot(0b01, t, Fraction(-1, 2))
        self._rot(0b10, t, Fraction(-1, 2))

    def ccz(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCZ gate to the given control and target qubits."""
        self._gadget([2, 0, 0], [c0, c1, t], Fraction(1, 4))
        self._gadget([0, 2, 0], [c0, c1, t], Fraction(1, 4))
        self._gadget([0, 0, 2], [c0, c1, t], Fraction(1, 4))
        self._gadget([0, 2, 2], [c0, c1, t], Fraction(-1, 4))
        self._gadget([2, 0, 2], [c0, c1, t], Fraction(-1, 4))
        self._gadget([2, 2, 0], [c0, c1, t], Fraction(-1, 4))
        self._gadget([2, 2, 2], [c0, c1, t], Fraction(1, 4))

    def ccy(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCY gate to the given control and target qubits."""
        self._rot(0b01, t, Fraction(1, 2))
        self.ccz(c0, c1, t)
        self._rot(0b01, t, Fraction(-1, 2))

    def cswap(self, c: QubitIdx, t0: QubitIdx, t1: QubitIdx) -> None:
        """Adds a CSWAP gate to the given control and target qubits."""
        self.cx(t1, t0)
        self.ccx(c, t0, t1)
        self.cx(t1, t0)

    if __debug__:

        @staticmethod
        def __validate_new_args(
            num_qubits: int, init_capacity: int, capacity_scaling: int | float
        ) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, SupportsIndex)
            validate(init_capacity, int)
            validate(capacity_scaling, int | float)
            num_qubits = int(num_qubits)
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            if init_capacity <= 0:
                raise ValueError("Circuit capacity must be >= 1.")
            if capacity_scaling <= 1.0:
                raise ValueError("Circuit capacity scalling must be > 1.")
            return True

        @staticmethod
        def __validate_qubit_idxs(q: QubitIdx | QubitIdxs) -> Literal[True]:
            if isinstance(q, QubitIdx):
                return True
            if isinstance(q, np.ndarray):
                if not np.issubdtype(q.dtype, np.unsignedinteger):
                    raise ValueError(
                        "Qubit indices specified by numpy arrays must be of uint dtype."
                    )
                return True
            validate(q, Sequence[int])
            if not all(_q >= 0 for _q in q):
                raise ValueError("Qubit indices must be >= 0.")
            if len(set(q)) != len(q):
                raise ValueError("Qubit indices must not be repeated.")
            return True

        def __validate_capacity(self, new_capacity: int) -> Literal[True]:
            if new_capacity <= 0:
                raise ValueError("Circuit capacity must be >= 1.")
            if new_capacity < self._num_gadgets:
                raise ValueError("Current number of gadgets exceeds desired capacity.")
            return True


# class CircuitBuilder(CircuitBuilderBase):
#     """Circuit builder where gadgets are stored in insertion order."""

#     _circuit: Circuit
#     _num_gadgets: int
#     _capacity_scaling: int | float

#     __slots__ = ("_circuit", "_num_gadgets", "_capacity_scaling")

#     def __new__(
#         cls,
#         num_qubits: int,
#         *,
#         init_capacity: int = 16,
#         capacity_scaling: int | float = 2,
#     ) -> Self:
#         self = super().__new__(cls, num_qubits)
#         assert CircuitBuilder.__validate_new_args(init_capacity, capacity_scaling)
#         self._circuit = Circuit.zero(init_capacity, num_qubits)
#         self._num_gadgets = 0
#         self._capacity_scaling = capacity_scaling
#         return self

#     @property
#     def capacity(self) -> int:
#         return len(self._circuit)

#     def append(self, gadget: Gadget) -> None:
#         if len(self) >= self.capacity:
#             self._scale_up_capacity(1)
#         self._circuit[len(self)] = gadget
#         self._num_gadgets += 1

#     def extend(self, gadgets: Sequence[Gadget] | Circuit) -> None:
#         if isinstance(gadgets, Circuit):
#             new_circuit = gadgets
#         else:
#             new_circuit = Circuit.from_gadgets(gadgets, self.num_qubits)
#         num_gadgets = len(self)
#         num_new_gadgets = len(new_circuit)
#         if num_gadgets + num_new_gadgets > self.capacity:
#             self._scale_up_capacity(num_new_gadgets)
#         self._circuit[num_gadgets : num_gadgets + num_new_gadgets] = new_circuit
#         self._num_gadgets += num_new_gadgets

#     def _scale_up_capacity(self, num_new_gadgets: int) -> None:
#         capacity = len(self._circuit) * 1.0
#         capacity_scaling = self._capacity_scaling
#         target_capacity = len(self) + num_new_gadgets
#         while capacity < target_capacity:
#             capacity *= capacity_scaling
#         self.set_capacity(int(ceil(capacity)))

#     def set_capacity(self, new_capacity: int) -> None:
#         """Sets the circuit capacity to the given value."""
#         assert self._validate_capacity(new_capacity)
#         circuit = self._circuit
#         capacity = len(circuit)
#         if new_capacity == capacity:
#             return
#         ext_circuit = Circuit.zero(new_capacity, self.num_qubits)
#         ext_circuit[:capacity] = circuit
#         self._circuit = ext_circuit

#     def trim_capacity(self) -> None:
#         """Sets the circuit capacity to the minimum amount possible."""
#         self.set_capacity(max(1, len(self)))

#     @override
#     def circuit(self) -> Circuit:
#         return self._circuit[: self._num_gadgets].clone()

#     def __iter__(self) -> Iterator[Gadget]:
#         yield from self._circuit[: self._num_gadgets]

#     def __len__(self) -> int:
#         return self._num_gadgets

#     def __repr__(self) -> str:
#         m, n = len(self), self.num_qubits
#         return f"<CircuitBuilder: {m} gadgets, {n} qubits>"

#     def __sizeof__(self) -> int:
#         return (
#             object.__sizeof__(self)
#             + self._num_qubits.__sizeof__()
#             + self._num_gadgets.__sizeof__()
#             + self._circuit.__sizeof__()
#         )

#     if __debug__:

#         @staticmethod
#         def __validate_new_args(
#             init_capacity: int, capacity_scaling: int | float
#         ) -> Literal[True]:
#             validate(init_capacity, int)
#             validate(capacity_scaling, int | float)
#             if init_capacity <= 0:
#                 raise ValueError("Circuit capacity must be >= 1.")
#             if capacity_scaling <= 1.0:
#                 raise ValueError("Circuit capacity scalling must be > 1.")
#             return True

#         def _validate_capacity(self, new_capacity: int) -> Literal[True]:
#             if new_capacity <= 0:
#                 raise ValueError("Circuit capacity must be >= 1.")
#             if new_capacity < self._num_gadgets:
#                 raise ValueError("Current number of gadgets exceeds desired capacity.")
#             return True


# class LayeredCircuitBuilder(CircuitBuilderBase):
#     """
#     Circuit builder where gadgets are fused into layers of
#     commuting gadgets with compatible legs.
#     """

#     _layers: list[Layer]

#     __slots__ = ("_layers",)

#     def __new__(cls, num_qubits: int) -> Self:
#         self = super().__new__(cls, num_qubits)
#         self._layers = []
#         return self

#     @property
#     def layers(self) -> Sequence[Layer]:
#         """Layers of the circuit."""
#         return tuple(self._layers)

#     @property
#     def num_layers(self) -> int:
#         """Number of layers in the circuit."""
#         return len(self._layers)

#     def append(self, gadget: Gadget) -> None:
#         assert self._validate_gadget(gadget)
#         m, n = self.num_layers, self._num_qubits
#         layers = self._layers
#         layer_idx = m
#         for i in range(m)[::-1]:
#             layer = layers[i]
#             if layer.is_compatible_with(gadget):
#                 layer_idx = i
#             elif not layer.commutes_with(gadget):
#                 break
#         if layer_idx < m:
#             layers[layer_idx].add_gadget(gadget)
#             return
#         new_layer = Layer(n)
#         new_layer.add_gadget(gadget)
#         layers.append(new_layer)

#     def extend(self, gadgets: Sequence[Gadget] | Circuit) -> None:
#         assert self._validate_gadgets(gadgets)
#         for gadget in gadgets:
#             self.append(gadget)

#     def __iter__(self) -> Iterator[Gadget]:
#         for layer in self._layers:
#             yield from layer

#     def __len__(self) -> int:
#         return sum(map(len, self._layers))

#     def random_circuit(self, *, rng: int | RNG | None) -> Circuit:
#         """
#         Returns a circuit constructed from the current gadget layers,
#         where the gadgets for each layer are listed in random order.
#         """
#         if not isinstance(rng, RNG):
#             rng = np.random.default_rng(rng)
#         return Circuit.from_gadgets(
#             g for layer in self._layers for g in rng.permutation(list(layer))  # type: ignore[arg-type]
#         )

#     def __repr__(self) -> str:
#         m, n = self.num_layers, self.num_qubits
#         return f"<LayeredCircuitBuilder: {m} layers, {n} qubits>"

#     def __sizeof__(self) -> int:
#         return (
#             object.__sizeof__(self)
#             + self._num_qubits.__sizeof__()
#             + self._layers.__sizeof__()
#             + sum(layer.__sizeof__() for layer in self._layers)
#         )
