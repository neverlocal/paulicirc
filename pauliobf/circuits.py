"""Circuits of Pauli gadgets."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Literal, Self, Sequence, SupportsIndex, final, overload
import numpy as np
from ._functions import (
    PHASE_NBYTES,
    RNG,
    GadgetCircData,
    GadgetData,
    Pauli,
    PauliArray,
    Phase,
    get_gadget_legs,
    get_phase,
    rand_circ,
    set_gadget_legs,
    set_phase,
    zero_circ,
)

if __debug__:
    from typing_validation import validate


@final
class Gadget:
    """A Pauli gadget."""

    _data: GadgetData
    _ephemeral: bool

    def __new__(cls, data: GadgetData, *, _ephemeral: bool = False) -> Self:
        """Constructs a Pauli gadget from the given data."""
        assert validate(data, GadgetData)
        self = super().__new__(cls)
        self._data = data
        self._ephemeral = _ephemeral
        return self

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the gadget."""
        return self._data.shape[0] - PHASE_NBYTES

    @property
    def legs(self) -> PauliArray:
        """Legs of the gadget."""
        return get_gadget_legs(self._data)

    @legs.setter
    def legs(self, value: Sequence[Pauli] | PauliArray) -> None:
        """Sets the legs of the gadget."""
        assert validate(value, Sequence[Pauli] | PauliArray)
        legs: PauliArray = np.asarray(value, dtype=np.uint8)
        assert self._validate_legs_value(legs)
        set_gadget_legs(self._data, legs)

    @property
    def phase(self) -> Phase:
        """Phase of the gadget."""
        return get_phase(self._data)

    @phase.setter
    def phase(self, value: Phase) -> None:
        """Sets the phase of the gadget."""
        assert validate(value, Phase)
        set_phase(self._data, value)

    def clone(self) -> Self:
        """Creates a persistent copy of the gadget."""
        return Gadget(self._data.copy())

    if __debug__:

        def _validate_legs_value(self, legs: PauliArray) -> Literal[True]:
            """Validates the value of the :attr:`legs` property."""
            if len(legs) != self.num_qubits:
                raise ValueError("Number of legs does not match number of qubits.")
            if not all(0 <= leg < 4 for leg in legs):
                raise ValueError("Legs must have value in range(4).")
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
        data = zero_circ(num_gadgets, num_qubits)
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
        data = rand_circ(num_gadgets, num_qubits, rng=rng)
        return cls(data)

    _data: GadgetCircData
    _num_qubits: int

    def __new__(cls, data: GadgetCircData, num_qubits: int | None = None) -> Self:
        """
        Constructs a gadget circuit from the given data.

        :meta public:
        """
        assert Circuit._validate_new_args(data, num_qubits)
        if num_qubits is None:
            num_qubits = data.shape[1] * 4
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

    def __iter__(self) -> Iterable[Gadget]:
        """
        Iterates over the gadgets in the circuit.

        .. warning::

            The gadgets yielded are ephemeral and should not be stored.
            Please use :meth:`Gadget.clone` to create a persistent copy.

        :meta public:
        """
        if len(self._data) == 0:
            return
        g = Gadget(self._data[0], _ephemeral=True)
        for row in self._data:
            g._data = row
            yield g

    @overload
    def __getitem__(self, idx: SupportsIndex) -> Gadget: ...
    @overload
    def __getitem__(self, idx: slice | Sequence[SupportsIndex]) -> Circuit: ...
    def __getitem__(
        self, idx: SupportsIndex | slice | Sequence[SupportsIndex]
    ) -> Gadget | Circuit:
        """
        Accesses the gadget at a given index, or selects/slices a sub-circuit.

        :meta public:
        """
        if isinstance(idx, slice):
            return Circuit(self._data[idx, :], self._num_qubits)
        if isinstance(idx, Sequence):
            assert validate(idx, Sequence[SupportsIndex])
            idx = list(idx)
            return Circuit(self._data[idx, :], self._num_qubits)  # type: ignore[index]
        assert validate(idx, SupportsIndex)
        idx = int(idx)
        return Gadget(self._data[idx])

    def __len__(self) -> int:
        """Number of gadgets in the circuit."""
        return len(self._data)

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
            data: GadgetCircData, num_qubits: int | None
        ) -> Literal[True]:
            """Validates the arguments of the :meth:`__new__` method."""
            validate(data, GadgetCircData)
            if num_qubits is not None:
                validate(num_qubits, int)
                if num_qubits < 0:
                    raise ValueError("Number of qubits must be non-negative.")
                if num_qubits > data.shape[1] * 4:
                    raise ValueError("Number of qubits exceeds circuit width.")
            return True
