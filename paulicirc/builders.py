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
from typing import (
    Any,
    Literal,
    Self,
    SupportsIndex,
    TypeAlias,
    overload,
    override,
    reveal_type,
)

import numpy as np

from ._numpy import RNG, Complex128Array1D, Complex128Array2D, normalise_phase
from .gadgets import PHASE_NBYTES, Gadget, Layer, PauliArray, Phase
from .circuits import Circuit, CircuitData

if __debug__:
    from typing_validation import validate

PhaseLike: TypeAlias = Phase | Fraction
r"""
Type alias for values which can be used to specify a phase:

- as a floating point value in :math:`[0, 2\pi)`, see :obj:`Phase`
- as a fraction of :math:`\pi`

"""

QubitIdx: TypeAlias = int
"""Type alias for the index of a qubit in a circuit."""


class CircuitBuilderBase(ABC):
    """
    Abstract base class for circuit builders,
    utility classes used to help building gadget circuits.
    """

    _num_qubits: int

    def __new__(cls, num_qubits: int) -> Self:
        """
        Create an empty circuit builder with the given number of qubits.

        :meta public:
        """
        assert CircuitBuilderBase._validate_new_args(num_qubits)
        self = super().__new__(cls)
        self._num_qubits = num_qubits
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits for the circuit."""
        return self._num_qubits

    @overload
    def add_gadget(
        self,
        phase: PhaseLike,
        legs: PauliArray,
        qubits: None = None,
    ) -> int: ...

    @overload
    def add_gadget(
        self,
        phase: PhaseLike,
        legs: str,
        qubits: QubitIdx | Sequence[QubitIdx] | None = None,
    ) -> int: ...

    def add_gadget(
        self,
        phase: PhaseLike,
        legs: PauliArray | str,
        qubits: QubitIdx | Sequence[QubitIdx] | None = None,
    ) -> int:
        """
        Add a gadget to the circuit.

        Returns the index of the layer to which the gadget was appended.
        """
        n = self._num_qubits
        if isinstance(phase, Phase):
            phase %= 2 * np.pi
        else:
            assert validate(phase, Fraction)
            phase = Gadget.frac2phase(phase)
        if isinstance(legs, str):
            paulis: str = legs
            PAULI_CHARS = "_XZY"
            if qubits is None:
                assert self._validate_gadget_args(legs, qubits)
                legs = np.fromiter(map(PAULI_CHARS.index, paulis), dtype=np.uint8)
            else:
                if isinstance(qubits, QubitIdx):
                    qubits = (qubits,)
                assert self._validate_gadget_args(legs, qubits)
                legs = np.zeros(n, dtype=np.uint8)
                for p, q in zip(paulis, qubits, strict=True):
                    legs[q] = PAULI_CHARS.index(p)
        return self._add_gadget(phase, legs)

    @abstractmethod
    def _add_gadget(self, phase: float, legs: PauliArray) -> int: ...

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
            res = gadget.unitary(_normalise_phase=False) @ res
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
            res = gadget.unitary(_normalise_phase=False) @ res
        if _normalise_phase:
            normalise_phase(res)
        return res

    def rz(self, angle: PhaseLike, q: QubitIdx) -> None:
        """Adds a Z rotation on the given qubit."""
        self.add_gadget(angle, "Z", q)

    def rx(self, angle: PhaseLike, q: QubitIdx) -> None:
        """Adds a Z rotation on the given qubit."""
        self.add_gadget(angle, "X", q)

    def ry(self, angle: PhaseLike, q: QubitIdx) -> None:
        """Adds a Y rotation on the given qubit."""
        self.add_gadget(angle, "Y", q)

    def z(self, q: QubitIdx) -> None:
        """Adds a Z gate on the given qubit."""
        self.rz(Fraction(1, 1), q)

    def x(self, q: QubitIdx) -> None:
        """Adds a X gate on the given qubit."""
        self.rx(Fraction(1, 1), q)

    def y(self, q: QubitIdx) -> None:
        """Adds a Y gate on the given qubit."""
        self.ry(Fraction(1, 1), q)

    def sx(self, q: QubitIdx) -> None:
        """Adds a √X gate on the given qubit."""
        self.rx(Fraction(1, 2), q)

    def sxdg(self, q: QubitIdx) -> None:
        """Adds a √X† gate on the given qubit."""
        self.rx(Fraction(-1, 2), q)

    def s(self, q: QubitIdx) -> None:
        """Adds a S gate on the given qubit."""
        self.rz(Fraction(1, 2), q)

    def sdg(self, q: QubitIdx) -> None:
        """Adds a S† gate on the given qubit."""
        self.rz(Fraction(-1, 2), q)

    def t(self, q: QubitIdx) -> None:
        """Adds a T gate on the given qubit."""
        self.rz(Fraction(1, 4), q)

    def tdg(self, q: QubitIdx) -> None:
        """Adds a T† gate on the given qubit."""
        self.rz(Fraction(-1, 4), q)

    def h(self, q: QubitIdx, *, xzx: bool = False) -> None:
        """
        Adds a H gate on the given qubit.

        By default, this is decomposed as ``Z(pi/2)X(pi/2)Z(pi/2)``,
        but setting ``xzx=True`` decomposes it as ``X(pi/2)Z(pi/2)X(pi/2)`` instead.
        """
        if xzx:
            self.sx(q)
            self.s(q)
            self.sx(q)
        else:
            self.s(q)
            self.sx(q)
            self.s(q)

    def hdg(self, q: QubitIdx, *, xzx: bool = False) -> None:
        """
        Adds a H gate on the given qubit.

        By default, this is decomposed as ``Z(-pi/2)X(-pi/2)Z(-pi/2)``,
        but setting ``xzx=True`` decomposes it as ``X(-pi/2)Z(-pi/2)X(-pi/2)`` instead.
        """
        if xzx:
            self.sxdg(q)
            self.sdg(q)
            self.sxdg(q)
        else:
            self.sdg(q)
            self.sxdg(q)
            self.sdg(q)

    def cz(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CZ gate to the given control and target qubits."""
        self.sdg(c)
        self.sdg(t)
        self.add_gadget(Fraction(1, 2), "ZZ", (c, t))

    def cx(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CX gate to the given control and target qubits."""
        self.s(t)
        self.sx(t)
        self.cz(c, t)
        self.sxdg(t)
        self.sdg(t)

    def cy(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CY gate to the given control and target qubits."""
        self.sx(t)
        self.cz(c, t)
        self.sxdg(t)

    def swap(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a SWAP gate to the given control and target qubits."""
        self.cx(c, t)
        self.cx(t, c)
        self.cx(c, t)

    def ccx(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCX gate to the given control and target qubits."""
        self.s(t)
        self.sx(t)
        self.ccz(c0, c1, t)
        self.sxdg(t)
        self.sdg(t)

    def ccz(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCZ gate to the given control and target qubits."""
        self.add_gadget(Fraction(1, 4), "Z__", (c0, c1, t))
        self.add_gadget(Fraction(1, 4), "_Z_", (c0, c1, t))
        self.add_gadget(Fraction(1, 4), "__Z", (c0, c1, t))
        self.add_gadget(Fraction(-1, 4), "ZZ_", (c0, c1, t))
        self.add_gadget(Fraction(-1, 4), "Z_Z", (c0, c1, t))
        self.add_gadget(Fraction(-1, 4), "_ZZ", (c0, c1, t))
        self.add_gadget(Fraction(1, 4), "ZZZ", (c0, c1, t))

    def ccy(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCY gate to the given control and target qubits."""
        self.sx(t)
        self.ccz(c0, c1, t)
        self.sxdg(t)

    def cswap(self, c: QubitIdx, t0: QubitIdx, t1: QubitIdx) -> None:
        """Adds a CSWAP gate to the given control and target qubits."""
        self.cx(t1, t0)
        self.ccx(c, t0, t1)
        self.cx(t1, t0)

    if __debug__:

        @staticmethod
        def _validate_new_args(num_qubits: int) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, SupportsIndex)
            num_qubits = int(num_qubits)
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            return True

        def _validate_gadget_args(
            self,
            legs: PauliArray | str,
            qubits: Sequence[QubitIdx] | None,
        ) -> Literal[True]:
            if qubits is None:
                if len(legs) != self.num_qubits:
                    raise ValueError(
                        "If qubits are not explicitly passed, the number of legs must "
                        "be exactly the number of qubits for the circuit builder."
                    )
            else:
                qubit_range = range(self.num_qubits)
                if not all(q in qubit_range for q in qubits):
                    raise ValueError(f"Qubit indices must fall in {qubit_range}")
                if len(legs) != len(qubits):
                    raise ValueError(
                        f"Found {len(legs)} legs, expected {len(qubits)} from qubits."
                    )
            if isinstance(legs, str):
                if not all(leg in "_XYZ" for leg in legs):
                    raise ValueError(
                        "Leg Pauli chars must be one of '_', 'X', 'Y' or 'Z'."
                    )
            else:
                if not all(0 <= leg_idx < 4 for leg_idx in legs):
                    raise ValueError("Leg Pauli indices must be in range(4).")
            return True


class CircuitBuilder(CircuitBuilderBase):
    """Circuit builder where gadgets are stored in insertion order."""

    _circuit: Circuit
    _num_gadgets: int

    def __new__(cls, num_qubits: int, *, init_capacity: int = 16) -> Self:
        self = super().__new__(cls, num_qubits)
        self._circuit = Circuit.zero(init_capacity, num_qubits)
        self._num_gadgets = 0
        return self

    def _add_gadget(self, phase: float, legs: PauliArray) -> int:
        idx = self._num_gadgets
        circuit = self._circuit
        gadget_data = Gadget.assemble_data(legs, phase)
        circuit._data[idx] = gadget_data
        self._num_gadgets += 1
        if idx + 1 == (capacity := len(circuit)):
            ext_circuit = Circuit.zero(2 * capacity, self.num_qubits)
            ext_circuit[:capacity] = circuit
            self._circuit = ext_circuit
        return idx

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


class LayeredCircuitBuilder(CircuitBuilderBase):
    """
    Circuit builder where gadgets are fused into layers of
    commuting gadgets with compatible legs.
    """

    _layers: list[Layer]

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

    def _add_gadget(self, phase: float, legs: PauliArray) -> int:
        m, n = self.num_layers, self._num_qubits
        layers = self._layers
        layer_idx = m
        for i in range(m)[::-1]:
            layer = layers[i]
            if layer.is_compatible_with(legs):
                layer_idx = i
            elif not layer.commutes_with(legs):
                break
        if layer_idx < m:
            layers[layer_idx].add_gadget(legs, phase)
            return layer_idx
        new_layer = Layer(n)
        new_layer.add_gadget(legs, phase)
        layers.append(new_layer)
        return m

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
