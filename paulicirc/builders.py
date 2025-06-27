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
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from fractions import Fraction
from math import ceil
from typing import (
    Literal,
    Self,
    SupportsIndex,
    TypeAlias,
    override,
)

import numpy as np

from .utils.numpy import RNG, Complex128Array1D, Complex128Array2D, normalise_phase
from .gadgets import (
    Gadget,
    Layer,
    Phase,
    QubitIdx,
    QubitIdxs,
    broadcast_idxs,
)
from .circuits import Circuit

if __debug__:
    from typing_validation import validate

PhaseLike: TypeAlias = Phase | Fraction
r"""
Type alias for values which can be used to specify a phase:

- as a floating point value in :math:`[0, 2\pi)`, see :obj:`Phase`
- as a fraction of :math:`\pi`

"""


class CircuitBuilderBase(ABC):
    """
    Abstract base class for circuit builders,
    utility classes used to help building gadget circuits.
    """

    _num_qubits: int

    __slots__ = ("__weakref__", "_num_qubits")

    def __new__(cls, num_qubits: int) -> Self:
        """
        Create an empty circuit builder with the given number of qubits.

        :meta public:
        """
        assert CircuitBuilderBase.__validate_new_args(num_qubits)
        self = super().__new__(cls)
        self._num_qubits = num_qubits
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits for the circuit."""
        return self._num_qubits

    # @overload
    # def add_gadget(
    #     self,
    #     phase: PhaseLike,
    #     legs: PauliArray,
    #     qubits: None = None,
    # ) -> None: ...

    # @overload
    # def add_gadget(
    #     self,
    #     phase: PhaseLike,
    #     legs: str,
    #     qubits: QubitIdx | Sequence[QubitIdx] | None = None,
    # ) -> None: ...

    # def add_gadget(
    #     self,
    #     phase: PhaseLike,
    #     legs: PauliArray | str,
    #     qubits: QubitIdx | Sequence[QubitIdx] | None = None,
    # ) -> None:
    #     """
    #     Add a gadget to the circuit.

    #     Returns the index of the layer to which the gadget was appended.
    #     """
    #     n = self._num_qubits
    #     if isinstance(phase, Phase):
    #         phase %= 2 * np.pi
    #     else:
    #         assert validate(phase, Fraction)
    #         phase = Gadget.frac2phase(phase)
    #     if isinstance(legs, str):
    #         paulis: str = legs
    #         PAULI_CHARS = "_XZY"
    #         if qubits is None:
    #             assert self._validate_gadget_args(legs, qubits)
    #             legs = np.fromiter(map(PAULI_CHARS.index, paulis), dtype=np.uint8)
    #         else:
    #             if isinstance(qubits, QubitIdx):
    #                 qubits = (qubits,)
    #             assert self._validate_gadget_args(legs, qubits)
    #             legs = np.zeros(n, dtype=np.uint8)
    #             for p, q in zip(paulis, qubits, strict=True):
    #                 legs[q] = PAULI_CHARS.index(p)
    #     self._add_gadget(phase, legs)

    @abstractmethod
    def append(self, gadget: Gadget) -> None:
        """Appends a single gadget to the circuit being built."""
        ...

    @abstractmethod
    def extend(self, gadgets: Sequence[Gadget] | Circuit) -> None:
        """
        Appends the given gadgets to the circuit being built.
        Gadgets are all validated prior to any modification,
        so either they are all appended to the circuit or none is.
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Gadget]:
        """Iterates over the gadgets in the ciruit builder."""

    @abstractmethod
    def __len__(self) -> int:
        """The number of gadgets currently in the circuit."""

    def circuit(self) -> Circuit:
        """Returns a circuit constructed from the gadgets currently in the builder."""
        return Circuit.from_gadgets(self, num_qubits=self._num_qubits)

    def unitary(self, *, _normalise_phase: bool = True) -> Complex128Array2D:
        """Returns the unitary matrix associated to the circuit being built."""
        res = np.eye(2**self.num_qubits, dtype=np.complex128)
        for gadget in self:
            res = gadget.unitary(normalize_phase=False) @ res
        if _normalise_phase:
            normalise_phase(res)
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
            res = gadget.unitary(normalize_phase=False) @ res
        if _normalise_phase:
            normalise_phase(res)
        return res

    def gadget(
        self, paulistr: str, qubits: QubitIdx | QubitIdxs, angle: PhaseLike
    ) -> None:
        """Adds a gadget with given Paulistring and angle on given qubits."""
        gadget = Gadget.from_sparse_paulistr(paulistr, qubits, self.num_qubits, angle)
        self.append(gadget)

    def rz(self, angle: PhaseLike, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Z rotation with given angle on the given qubit(s)."""
        (qs,) = broadcast_idxs(q)
        for q in qs:
            self.gadget("Z", q, angle)

    def rx(self, angle: PhaseLike, q: QubitIdx | QubitIdxs) -> None:
        """Adds an X rotation with given angle on the given qubit(s)."""
        (qs,) = broadcast_idxs(q)
        for q in qs:
            self.gadget("X", q, angle)

    def ry(self, angle: PhaseLike, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Y rotation with given angle on the given qubit(s)."""
        (qs,) = broadcast_idxs(q)
        for q in qs:
            self.gadget("Y", q, angle)

    def z(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Z gate on the given qubit."""
        self.rz(Fraction(1, 1), q)

    def x(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a X gate on the given qubit."""
        self.rx(Fraction(1, 1), q)

    def y(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a Y gate on the given qubit."""
        self.ry(Fraction(1, 1), q)

    def sx(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a √X gate on the given qubit."""
        self.rx(Fraction(1, 2), q)

    def sxdg(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a √X† gate on the given qubit."""
        self.rx(Fraction(-1, 2), q)

    def s(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a S gate on the given qubit."""
        self.rz(Fraction(1, 2), q)

    def sdg(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a S† gate on the given qubit."""
        self.rz(Fraction(-1, 2), q)

    def t(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a T gate on the given qubit."""
        self.rz(Fraction(1, 4), q)

    def tdg(self, q: QubitIdx | QubitIdxs) -> None:
        """Adds a T† gate on the given qubit."""
        self.rz(Fraction(-1, 4), q)

    def h(self, q: QubitIdx | QubitIdxs, *, xzx: bool = False) -> None:
        """
        Adds a H gate on the given qubit.

        By default, this is decomposed as ``Z(pi/2)X(pi/2)Z(pi/2)``,
        but setting ``xzx=True`` decomposes it as ``X(pi/2)Z(pi/2)X(pi/2)`` instead.
        """
        (qs,) = broadcast_idxs(q)
        if xzx:
            for q in qs:
                self.sx(q)
                self.s(q)
                self.sx(q)
        else:
            for q in qs:
                self.s(q)
                self.sx(q)
                self.s(q)

    def hdg(self, q: QubitIdx | QubitIdxs, *, xzx: bool = False) -> None:
        """
        Adds a H gate on the given qubit.

        By default, this is decomposed as ``Z(-pi/2)X(-pi/2)Z(-pi/2)``,
        but setting ``xzx=True`` decomposes it as ``X(-pi/2)Z(-pi/2)X(-pi/2)`` instead.
        """
        (qs,) = broadcast_idxs(q)
        if xzx:
            for q in qs:
                self.sxdg(q)
                self.sdg(q)
                self.sxdg(q)
        else:
            for q in qs:
                self.sdg(q)
                self.sxdg(q)
                self.sdg(q)

    def cz(self, c: QubitIdx | QubitIdxs, t: QubitIdx | QubitIdxs) -> None:
        """Adds a CZ gate to the given control and target qubits."""
        cs, ts = broadcast_idxs(c, t)
        for c, t in zip(cs, ts):
            self.sdg(c)
            self.sdg(t)
            self.gadget("ZZ", (c, t), Fraction(1, 2))

    def cx(self, c: QubitIdx | QubitIdxs, t: QubitIdx | QubitIdxs) -> None:
        """Adds a CX gate to the given control and target qubits."""
        cs, ts = broadcast_idxs(c, t)
        for c, t in zip(cs, ts):
            self.s(t)
            self.sx(t)
            self.cz(c, t)
            self.sxdg(t)
            self.sdg(t)

    def cy(self, c: QubitIdx | QubitIdxs, t: QubitIdx | QubitIdxs) -> None:
        """Adds a CY gate to the given control and target qubits."""
        cs, ts = broadcast_idxs(c, t)
        for c, t in zip(cs, ts):
            self.sx(t)
            self.cz(c, t)
            self.sxdg(t)

    def swap(self, c: QubitIdx | QubitIdxs, t: QubitIdx | QubitIdxs) -> None:
        """Adds a SWAP gate to the given control and target qubits."""
        cs, ts = broadcast_idxs(c, t)
        for c, t in zip(cs, ts):
            self.cx(c, t)
            self.cx(t, c)
            self.cx(c, t)

    def ccx(
        self,
        c0: QubitIdx | QubitIdxs,
        c1: QubitIdx | QubitIdxs,
        t: QubitIdx | QubitIdxs,
    ) -> None:
        """Adds a CCX gate to the given control and target qubits."""
        c0s, c1s, ts = broadcast_idxs(c0, c1, t)
        for c0, c1, t in zip(c0s, c1s, ts):
            self.s(t)
            self.sx(t)
            self.ccz(c0, c1, t)
            self.sxdg(t)
            self.sdg(t)

    def ccz(
        self,
        c0: QubitIdx | QubitIdxs,
        c1: QubitIdx | QubitIdxs,
        t: QubitIdx | QubitIdxs,
    ) -> None:
        """Adds a CCZ gate to the given control and target qubits."""
        c0s, c1s, ts = broadcast_idxs(c0, c1, t)
        for c0, c1, t in zip(c0s, c1s, ts):
            self.gadget("Z__", (c0, c1, t), Fraction(1, 4))
            self.gadget("_Z_", (c0, c1, t), Fraction(1, 4))
            self.gadget("__Z", (c0, c1, t), Fraction(1, 4))
            self.gadget("_ZZ", (c0, c1, t), Fraction(-1, 4))
            self.gadget("Z_Z", (c0, c1, t), Fraction(-1, 4))
            self.gadget("ZZ_", (c0, c1, t), Fraction(-1, 4))
            self.gadget("ZZZ", (c0, c1, t), Fraction(1, 4))

    def ccy(
        self,
        c0: QubitIdx | QubitIdxs,
        c1: QubitIdx | QubitIdxs,
        t: QubitIdx | QubitIdxs,
    ) -> None:
        """Adds a CCY gate to the given control and target qubits."""
        c0s, c1s, ts = broadcast_idxs(c0, c1, t)
        for c0, c1, t in zip(c0s, c1s, ts):
            self.sx(t)
            self.ccz(c0, c1, t)
            self.sxdg(t)

    def cswap(
        self,
        c: QubitIdx | QubitIdxs,
        t0: QubitIdx | QubitIdxs,
        t1: QubitIdx | QubitIdxs,
    ) -> None:
        """Adds a CSWAP gate to the given control and target qubits."""
        cs, t0s, t1s = broadcast_idxs(c, t0, t1)
        for c, t0, t1 in zip(cs, t0s, t1s):
            self.cx(t1, t0)
            self.ccx(c, t0, t1)
            self.cx(t1, t0)

    if __debug__:

        @staticmethod
        def __validate_new_args(num_qubits: int) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, SupportsIndex)
            num_qubits = int(num_qubits)
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            return True

        def _validate_gadget(self, gadget: Gadget) -> Literal[True]:
            validate(gadget, Gadget)
            if gadget.num_qubits != self.num_qubits:
                raise ValueError(
                    f"Found {gadget.num_qubits} qubits, expected {self.num_qubits}."
                )
            return True

        def _validate_gadgets(
            self, gadgets: Sequence[Gadget] | Circuit
        ) -> Literal[True]:
            validate(gadgets, Sequence[Gadget] | Circuit)
            num_qubits = self.num_qubits
            for idx, gadget in enumerate(gadgets):
                if gadget.num_qubits != num_qubits:
                    raise ValueError(
                        f"Found {gadget.num_qubits} qubits at {idx = }"
                        f", expected {num_qubits}."
                    )
            return True


class CircuitBuilder(CircuitBuilderBase):
    """Circuit builder where gadgets are stored in insertion order."""

    _circuit: Circuit
    _num_gadgets: int
    _capacity_scaling: int | float

    __slots__ = ("_circuit", "_num_gadgets", "_capacity_scaling")

    def __new__(
        cls,
        num_qubits: int,
        *,
        init_capacity: int = 16,
        capacity_scaling: int | float = 2,
    ) -> Self:
        self = super().__new__(cls, num_qubits)
        assert CircuitBuilder.__validate_new_args(init_capacity, capacity_scaling)
        self._circuit = Circuit.zero(init_capacity, num_qubits)
        self._num_gadgets = 0
        self._capacity_scaling = capacity_scaling
        return self

    @property
    def capacity(self) -> int:
        return len(self._circuit)

    def append(self, gadget: Gadget) -> None:
        if len(self) >= self.capacity:
            self._scale_up_capacity(1)
        self._circuit[len(self)] = gadget
        self._num_gadgets += 1

    def extend(self, gadgets: Sequence[Gadget] | Circuit) -> None:
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
        assert self._validate_capacity(new_capacity)
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

    @override
    def circuit(self) -> Circuit:
        return self._circuit[: self._num_gadgets].clone()

    def __iter__(self) -> Iterator[Gadget]:
        yield from self._circuit[: self._num_gadgets]

    def __len__(self) -> int:
        return self._num_gadgets

    def __repr__(self) -> str:
        m, n = len(self), self.num_qubits
        return f"<CircuitBuilder: {m} gadgets, {n} qubits>"

    def __sizeof__(self) -> int:
        return (
            object.__sizeof__(self)
            + self._num_qubits.__sizeof__()
            + self._num_gadgets.__sizeof__()
            + self._circuit.__sizeof__()
        )

    if __debug__:

        @staticmethod
        def __validate_new_args(
            init_capacity: int, capacity_scaling: int | float
        ) -> Literal[True]:
            validate(init_capacity, int)
            validate(capacity_scaling, int | float)
            if init_capacity <= 0:
                raise ValueError("Circuit capacity must be >= 1.")
            if capacity_scaling <= 1.0:
                raise ValueError("Circuit capacity scalling must be > 1.")
            return True

        def _validate_capacity(self, new_capacity: int) -> Literal[True]:
            if new_capacity <= 0:
                raise ValueError("Circuit capacity must be >= 1.")
            if new_capacity < self._num_gadgets:
                raise ValueError("Current number of gadgets exceeds desired capacity.")
            return True


class LayeredCircuitBuilder(CircuitBuilderBase):
    """
    Circuit builder where gadgets are fused into layers of
    commuting gadgets with compatible legs.
    """

    _layers: list[Layer]

    __slots__ = ("_layers",)

    def __new__(cls, num_qubits: int) -> Self:
        self = super().__new__(cls, num_qubits)
        self._layers = []
        return self

    @property
    def layers(self) -> Sequence[Layer]:
        """Layers of the circuit."""
        return tuple(self._layers)

    @property
    def num_layers(self) -> int:
        """Number of layers in the circuit."""
        return len(self._layers)

    def append(self, gadget: Gadget) -> None:
        assert self._validate_gadget(gadget)
        m, n = self.num_layers, self._num_qubits
        layers = self._layers
        layer_idx = m
        for i in range(m)[::-1]:
            layer = layers[i]
            if layer.is_compatible_with(gadget):
                layer_idx = i
            elif not layer.commutes_with(gadget):
                break
        if layer_idx < m:
            layers[layer_idx].add_gadget(gadget)
            return
        new_layer = Layer(n)
        new_layer.add_gadget(gadget)
        layers.append(new_layer)

    def extend(self, gadgets: Sequence[Gadget] | Circuit) -> None:
        assert self._validate_gadgets(gadgets)
        for gadget in gadgets:
            self.append(gadget)

    def __iter__(self) -> Iterator[Gadget]:
        for layer in self._layers:
            yield from layer

    def __len__(self) -> int:
        return sum(map(len, self._layers))

    def random_circuit(self, *, rng: int | RNG | None) -> Circuit:
        """
        Returns a circuit constructed from the current gadget layers,
        where the gadgets for each layer are listed in random order.
        """
        if not isinstance(rng, RNG):
            rng = np.random.default_rng(rng)
        return Circuit.from_gadgets(
            g for layer in self._layers for g in rng.permutation(list(layer))  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        m, n = self.num_layers, self.num_qubits
        return f"<LayeredCircuitBuilder: {m} layers, {n} qubits>"

    def __sizeof__(self) -> int:
        return (
            object.__sizeof__(self)
            + self._num_qubits.__sizeof__()
            + self._layers.__sizeof__()
            + sum(layer.__sizeof__() for layer in self._layers)
        )
