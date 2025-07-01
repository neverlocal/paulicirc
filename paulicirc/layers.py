"""Layers of Pauli gadgets."""

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
    final,
    overload,
)
import numpy as np
from .gadgets import PAULI_CHARS, Gadget, PauliArray, Phase, are_same_phase
from .circuits import Circuit, statevec_from_gadgets, unitary_from_gadgets
from .utils.numpy import RNG, Complex128Array1D, Complex128Array2D, ComplexArray1D, FloatArray1D, canonicalize_phase

if __debug__:
    from typing_validation import validate


@final
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
        assert self.__validate_legs_self(legs)
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
        assert self.__validate_legs_self(legs)
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

    def unitary(
        self,
        *,
        canonical_phase: bool = True,
        _use_cupy: bool = False,  # currently in alpha
    ) -> Complex128Array2D:
        """Returns the unitary matrix associated to this Pauli gadget circuit."""
        return unitary_from_gadgets(self, self.num_qubits, canonical_phase, _use_cupy)

    def statevec(
        self,
        input: ComplexArray1D | FloatArray1D,
        canonical_phase: bool = True,
        _use_cupy: bool = False,  # currently in alpha
    ) -> Complex128Array1D:
        """
        Computes the statevector resulting from the application of this gadget circuit
        to the given input statevector.
        """
        return statevec_from_gadgets(self, input, canonical_phase, _use_cupy)

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

        def __validate_legs_self(self, legs: PauliArray) -> Literal[True]:
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


@final
class LayeredCircuit:
    """A quantum circuit, represented as a sequential composition of Pauli layers."""

    _num_qubits: int
    _layers: list[Layer]

    __slots__ = (
        "__weakref__",
        "_num_qubits",
        "_layers",
    )

    def __new__(cls, num_qubits: int) -> Self:
        assert LayeredCircuit.__validate_new_args(num_qubits)
        self = super().__new__(cls)
        self._num_qubits = num_qubits
        self._layers = []
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._num_qubits

    @property
    def layers(self) -> Sequence[Layer]:
        """Layers of the circuit."""
        return tuple(self._layers)

    @property
    def num_layers(self) -> int:
        """Number of layers in the circuit."""
        return len(self._layers)

    def append(self, gadget: Gadget) -> None:
        """Appends a gadget to the layered circuit."""
        assert self.__validate_gadget(gadget)
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

    def extend(self, gadgets: Iterable[Gadget]) -> None:
        """Appends a sequence of gadgets to the layered circuit."""
        for gadget in gadgets:
            self.append(gadget)

    def unitary(
        self,
        *,
        canonical_phase: bool = True,
        _use_cupy: bool = False,  # currently in alpha
    ) -> Complex128Array2D:
        """Returns the unitary matrix associated to this Pauli gadget circuit."""
        return unitary_from_gadgets(self, self.num_qubits, canonical_phase, _use_cupy)

    def statevec(
        self,
        input: ComplexArray1D | FloatArray1D,
        canonical_phase: bool = True,
        _use_cupy: bool = False,  # currently in alpha
    ) -> Complex128Array1D:
        """
        Computes the statevector resulting from the application of this gadget circuit
        to the given input statevector.
        """
        return statevec_from_gadgets(self, input, canonical_phase, _use_cupy)


    def __iter__(self) -> Iterator[Gadget]:
        for layer in self._layers:
            yield from layer

    def __len__(self) -> int:
        return sum(map(len, self._layers))

    def circuit(self) -> Circuit:
        """
        Returns a circuit constructed from the current gadget layers,
        where the gadgets for each layer are listed in canonical order.
        """
        return Circuit.from_gadgets(self, self.num_qubits)

    def random_circuit(self, *, rng: int | RNG | None) -> Circuit:
        """
        Returns a circuit constructed from the current gadget layers,
        where the gadgets for each layer are listed in random order.
        """
        if not isinstance(rng, RNG):
            rng = np.random.default_rng(rng)
        return Circuit.from_gadgets(
            (g for layer in self._layers for g in rng.permutation(list(layer))),  # type: ignore[arg-type]
            self.num_qubits,
        )

    def __repr__(self) -> str:
        m, n = self.num_layers, self.num_qubits
        return f"<LayeredCircuit: {m} layers, {n} qubits>"

    def __sizeof__(self) -> int:
        return (
            object.__sizeof__(self)
            + self._num_qubits.__sizeof__()
            + self._layers.__sizeof__()
            + sum(layer.__sizeof__() for layer in self._layers)
        )

    if __debug__:

        @staticmethod
        def __validate_new_args(
            num_qubits: int,
        ) -> Literal[True]:
            """Validate arguments to the :meth:`__new__` method."""
            validate(num_qubits, SupportsIndex)
            if num_qubits < 0:
                raise ValueError("Number of qubits must be non-negative.")
            return True

        def __validate_gadget(self, gadget: Gadget) -> Literal[True]:
            validate(gadget, Gadget)
            if gadget.num_qubits != self.num_qubits:
                raise ValueError(
                    f"Found {gadget.num_qubits} qubits, expected {self.num_qubits}."
                )
            return True
