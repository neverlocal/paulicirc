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
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal, Self, TypeAlias

import numpy as np
from .gadgets import PHASE_DENOM, Gadget, Layer, PauliArray, Phase

if __debug__:
    from typing_validation import validate

PhaseLike: TypeAlias = Phase | float | Fraction
r"""
Type alias for values which can be used to specify a phase:

- as an integer, see :obj:`Phase`
- as a floating point value in :math:`[0, 2\pi)`
- as a fraction of :math:`\pi`

"""

QubitIdx: TypeAlias = int
"""Type alias for the index of a qubit in a circuit."""

class CircuitBuilder:
    """Utility class to help building gadget circuits."""

    _num_qubits: int
    _layers: list[Layer]

    def __new__(cls, num_qubits: int) -> Self:
        """
        Create an empty circuit builder with the given number of qubits.

        :meta public:
        """
        assert CircuitBuilder._validate_new_args(num_qubits)
        self = super().__new__(cls)
        self._num_qubits = num_qubits
        self._layers = []
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits for the circuit."""
        return self._num_qubits

    @property
    def layers(self) -> Sequence[Layer]:
        """Layers of the circuit."""
        return tuple(self._layers)

    @property
    def num_layers(self) -> int:
        """Number of layers in the circuit."""
        return len(self._layers)

    def add_gadget(
        self,
        phase: PhaseLike,
        legs: PauliArray | str,
        qubits: QubitIdx | Sequence[QubitIdx] | None = None
    ) -> int:
        """
        Add a gadget to the circuit.

        Returns the index of the layer to which the gadget was appended.
        """
        if isinstance(phase, Phase):
            phase %= PHASE_DENOM
        elif isinstance(phase, float):
            phase = Gadget.float2phase(phase)
        else:
            assert validate(phase, Fraction)
            phase = Gadget.frac2phase(phase)
        if isinstance(legs, str):
            paulis: str = legs
            PAULI_CHARS = "_XZY"
            if qubits is None:
                # TODO: validate legs
                legs = np.fromiter(map(PAULI_CHARS.index, paulis), dtype=np.uint8)
            else:
                if isinstance(qubits, QubitIdx):
                    qubits = (qubits,)
                # TODO: validate legs and qubits
                legs = np.zeros(self._num_qubits, dtype=np.uint8)
                for p, q in zip(paulis, qubits, strict=True):
                    legs[q] = PAULI_CHARS.index(p)
        layers = self._layers
        for i in range(len(self._layers))[::-1]:
            layer = layers[i]
            if layer.add_gadget(legs, phase):
                return i
            if not layer.commutes_with(legs):
                break
        new_layer = Layer(self._num_qubits)
        new_layer.add_gadget(legs, phase)
        layers.append(new_layer)
        return len(layers) - 1

    def rx(self, angle: PhaseLike, q: QubitIdx) -> None:
        """Adds a Z rotation on the given qubit."""
        self.add_gadget(angle, "X", q)

    def rz(self, angle: PhaseLike, q: QubitIdx) -> None:
        """Adds a Z rotation on the given qubit."""
        self.add_gadget(angle, "Z", q)

    def ry(self, angle: PhaseLike, q: QubitIdx) -> None:
        """Adds a Y rotation on the given qubit."""
        self.add_gadget(angle, "Y", q)

    def x(self, q: QubitIdx) -> None:
        """Adds a X gate on the given qubit."""
        self.rx(PHASE_DENOM//2, q)

    def z(self, q: QubitIdx) -> None:
        """Adds a Z gate on the given qubit."""
        self.rz(PHASE_DENOM//2, q)

    def y(self, q: QubitIdx) -> None:
        """Adds a Y gate on the given qubit."""
        self.ry(PHASE_DENOM//2, q)

    def sx(self, q: QubitIdx) -> None:
        """Adds a √X gate on the given qubit."""
        self.rx(PHASE_DENOM//4, q)

    def sx_dag(self, q: QubitIdx) -> None:
        """Adds a √X† gate on the given qubit."""
        self.rx(-PHASE_DENOM//4, q)

    def s(self, q: QubitIdx) -> None:
        """Adds a S gate on the given qubit."""
        self.rz(PHASE_DENOM//4, q)

    def s_dag(self, q: QubitIdx) -> None:
        """Adds a S† gate on the given qubit."""
        self.rz(-PHASE_DENOM//4, q)

    def t(self, q: QubitIdx) -> None:
        """Adds a T gate on the given qubit."""
        self.rz(PHASE_DENOM//8, q)

    def t_dag(self, q: QubitIdx) -> None:
        """Adds a T† gate on the given qubit."""
        self.rz(-PHASE_DENOM//48, q)

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

    def h_dag(self, q: QubitIdx, *, xzx: bool = False) -> None:
        """
        Adds a H gate on the given qubit.

        By default, this is decomposed as ``Z(-pi/2)X(-pi/2)Z(-pi/2)``,
        but setting ``xzx=True`` decomposes it as ``X(-pi/2)Z(-pi/2)X(-pi/2)`` instead.
        """
        if xzx:
            self.sx_dag(q)
            self.s_dag(q)
            self.sx_dag(q)
        else:
            self.s_dag(q)
            self.sx_dag(q)
            self.s_dag(q)

    def cx(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CX gate to the given control and target qubits."""
        self.s(t)
        self.sx(t)
        self.cz(c, t)
        self.sx_dag(t)
        self.s_dag(t)

    def cz(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CZ gate to the given control and target qubits."""
        self.s_dag(c)
        self.s_dag(t)
        self.add_gadget(PHASE_DENOM//4, "ZZ", (c, t))

    def cy(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a CY gate to the given control and target qubits."""
        self.sx(t)
        self.cz(c, t)
        self.sx(t)

    def swap(self, c: QubitIdx, t: QubitIdx) -> None:
        """Adds a SWAP gate to the given control and target qubits."""
        # Simplified from cx(c, t) >> cx(t, c) >> cx(c, t)
        self.s(t)
        self.sx(t)
        self.s_dag(t)
        self.add_gadget(PHASE_DENOM//4, "ZZ", (c, t))
        self.sx(c)
        self.sx_dag(t)
        self.s_dag(c)
        self.s_dag(t)
        self.add_gadget(PHASE_DENOM//4, "ZZ", (c, t))
        self.sx_dag(c)
        self.sx(t)
        self.z(c)
        self.s_dag(t)
        self.add_gadget(PHASE_DENOM//4, "ZZ", (c, t))
        self.s_dag(t)
        self.s(t)

    def ccx(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCX gate to the given control and target qubits."""
        self.s(t)
        self.sx(t)
        self.ccz(c0, c1, t)
        self.sx(t)
        self.s(t)

    def ccz(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCZ gate to the given control and target qubits."""
        self.add_gadget(PHASE_DENOM//8, "Z__", (c0, c1, t))
        self.add_gadget(PHASE_DENOM//8, "_Z_", (c0, c1, t))
        self.add_gadget(PHASE_DENOM//8, "__Z", (c0, c1, t))
        self.add_gadget(-PHASE_DENOM//8, "ZZ_", (c0, c1, t))
        self.add_gadget(-PHASE_DENOM//8, "Z_Z", (c0, c1, t))
        self.add_gadget(-PHASE_DENOM//8, "_ZZ", (c0, c1, t))
        self.add_gadget(PHASE_DENOM//8, "ZZZ", (c0, c1, t))

    def ccy(self, c0: QubitIdx, c1: QubitIdx, t: QubitIdx) -> None:
        """Adds a CCY gate to the given control and target qubits."""
        self.sx(t)
        self.ccz(c0, c1, t)
        self.sx(t)

    def cswap(self, c: QubitIdx, t0: QubitIdx, t1: QubitIdx) -> None:
        """Adds a CSWAP gate to the given control and target qubits."""
        self.s(t0)
        self.sx(t0)
        self.cz(t0, t1)
        self.sx_dag(t0)
        self.ccx(c, t0, t1)
        self.sx(t0)
        self.cz(t0, t1)
        self.sx_dag(t0)
        self.s_dag(t0)

    if __debug__:

        @staticmethod
        def _validate_new_args(num_qubits: int) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, int)
            if num_qubits <= 0:
                raise ValueError("Number of qubits must be strictly positive.")
            return True
