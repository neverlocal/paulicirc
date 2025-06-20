from collections.abc import Sequence, Set
import numpy as np
import pytest

from paulicirc.gadgets import (
    Gadget,
    Layer,
    PauliArray,
    Phase,
    is_zero_phase,
)

RNG_SEED = 0
RNG_ALT_SEED = 1
NUM_RNG_SAMPLES = 10
NUM_QUBITS_RANGE = range(0, 9)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize("num_qubits", NUM_QUBITS_RANGE)
def test_empty_layer(num_qubits: int) -> None:
    layer = Layer(num_qubits)
    assert layer.num_qubits == num_qubits
    assert len(layer) == 0
    assert list(layer) == []


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,gadget",
    [
        (num_qubits, Gadget.random(num_qubits, rng=rng, allow_zero=False))
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_layer_single_gadget(num_qubits: int, gadget: Gadget) -> None:
    layer = Layer(num_qubits)
    layer.add_gadget(gadget)
    assert layer.num_qubits == gadget.num_qubits == num_qubits
    assert len(layer) == 1
    assert np.array_equal(layer.legs, gadget.legs)
    assert layer.phase(gadget.legs) == gadget.phase
    assert list(layer) == [gadget]


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,legs,phases",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.uniform(0, 2 * np.pi, size=rng.integers(2, 5)),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_layer_same_legs(
    num_qubits: int, legs: PauliArray, phases: Sequence[Phase]
) -> None:
    layer = Layer(num_qubits)
    overall_phase = 0.0
    for phase in phases:
        overall_phase += phase
        layer.add_gadget(legs, phase)
    assert layer.num_qubits == num_qubits
    assert np.array_equal(layer.legs, legs)
    assert len(layer) == 1
    assert is_zero_phase(layer.phase(legs) - overall_phase)
    assert list(layer) == [Gadget.from_legs(legs, overall_phase)]


rng = np.random.default_rng(RNG_SEED)
NUM_SUBSETS = 20


def select_leg_subset(subset: Set[int], legs: PauliArray) -> PauliArray:
    selected_legs = legs.copy()
    for q in range(len(legs)):
        if q not in subset:
            selected_legs[q] = 0
    return selected_legs


# TODO: separately test that Layer.select_leg_subset matches select_leg_subset


@pytest.mark.parametrize(
    "num_qubits,legs,subsets,phases",
    [
        (
            num_qubits,
            rng.integers(1, 4, size=num_qubits, dtype=np.uint8),
            [
                frozenset(
                    map(
                        int,
                        rng.choice(
                            num_qubits,
                            replace=False,
                            size=rng.integers(0, num_qubits + 1),
                        ),
                    )
                )
                for _ in range(NUM_SUBSETS)
            ],
            rng.uniform(0, 2 * np.pi / (NUM_SUBSETS + 1), size=NUM_SUBSETS),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_layer_different_legs(
    num_qubits: int,
    legs: PauliArray,
    subsets: Sequence[frozenset[int]],
    phases: Sequence[Phase],
) -> None:
    layer = Layer(num_qubits)
    union_subset: set[int] = set()
    unique_subsets: list[frozenset[int]] = []
    gadgets: dict[frozenset[int], Gadget] = {}
    for subset, phase in zip(subsets, phases, strict=True):
        subset_legs = select_leg_subset(subset, legs)
        assert np.array_equal(subset_legs, Layer.select_leg_subset(subset, legs))
        union_subset |= subset
        layer.add_gadget(subset_legs, phase)
        if subset in unique_subsets:
            gadgets[subset].phase += phase
        else:
            g = Gadget.from_legs(subset_legs, phase)
            unique_subsets.append(subset)
            gadgets[subset] = g
    covered_legs = select_leg_subset(union_subset, legs)
    assert layer.num_qubits == num_qubits
    assert np.array_equal(layer.legs, covered_legs)
    assert len(layer) == len(unique_subsets)
    assert list(layer) == [gadgets[subset] for subset in unique_subsets]


@pytest.mark.parametrize(
    "num_qubits,legs,subsets,phases",
    [
        (
            num_qubits,
            rng.integers(1, 4, size=num_qubits, dtype=np.uint8),
            [
                frozenset(
                    map(
                        int,
                        rng.choice(
                            num_qubits,
                            replace=False,
                            size=rng.integers(0, num_qubits + 1),
                        ),
                    )
                )
                for _ in range(NUM_SUBSETS)
            ],
            rng.uniform(0, 2 * np.pi / (NUM_SUBSETS + 1), size=NUM_SUBSETS),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_layer_cancelling_phases(
    num_qubits: int,
    legs: PauliArray,
    subsets: Sequence[frozenset[int]],
    phases: Sequence[Phase],
) -> None:
    layer = Layer(num_qubits)
    union_subset: set[int] = set()
    unique_subsets: list[frozenset[int]] = []
    gadgets: dict[frozenset[int], Gadget] = {}
    for subset, phase in zip(subsets, phases, strict=True):
        subset_legs = Layer.select_leg_subset(subset, legs)
        union_subset |= subset
        layer.add_gadget(subset_legs, phase)
        if subset in unique_subsets:
            gadgets[subset].phase += phase
        else:
            g = Gadget.from_legs(subset_legs, phase)
            unique_subsets.append(subset)
            gadgets[subset] = g
    for subset in unique_subsets:
        subset_legs = Layer.select_leg_subset(subset, legs)
        phase = layer.phase(subset_legs)
        layer.add_gadget(subset_legs, -phase)
    assert layer.num_qubits == num_qubits
    assert np.all(layer.legs == 0)
    assert len(layer) == 0
    assert list(layer) == []


@pytest.mark.parametrize(
    "num_qubits,legs_fst,legs_snd",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_layer_incompatibility(
    num_qubits: int,
    legs_fst: PauliArray,
    legs_snd: PauliArray,
) -> None:
    legs_compatible = all(
        leg_fst == 0 or leg_snd == 0 or leg_fst == leg_snd
        for leg_fst, leg_snd in zip(legs_fst, legs_snd, strict=True)
    )
    layer = Layer(num_qubits)
    layer.add_gadget(legs_fst, np.pi / 2)
    assert layer.add_gadget(legs_snd, np.pi / 2) == legs_compatible


# Unitary?
