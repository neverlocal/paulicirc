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
from typing import Literal, Self

from .gadgets import Layer, PauliArray, Phase

if __debug__:
    from typing_validation import validate


class CircuitBuilder:

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

    def add_gadget(self, legs: PauliArray, phase: Phase) -> int:
        """Add a phase to the layer."""
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

    if __debug__:

        @staticmethod
        def _validate_new_args(num_qubits: int) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, int)
            if num_qubits <= 0:
                raise ValueError("Number of qubits must be strictly positive.")
            return True
