import numpy as np
import pytest

from pauliobf.gadgets import (
    PAULI_CHARS,
    PHASE_NBYTES,
    Gadget,
    PauliArray,
    Phase,
    is_zero_phase,
    set_phase,
    zero_gadget_data,
    get_gadget_legs,
    set_gadget_legs,
    get_phase,
)

RNG_SEED = 0
RNG_ALT_SEED = 1
NUM_RNG_SAMPLES = 10
NUM_QUBITS_RANGE = range(0, 9)


def num_leg_bytes(num_qubits: int) -> int:
    return -(-num_qubits // 4)


def round_num_legs(num_qubits: int) -> int:
    return num_leg_bytes(num_qubits) * 4


@pytest.mark.parametrize("num_qubits", NUM_QUBITS_RANGE)
def test_zero_gadget_np(num_qubits: int) -> None:
    data = zero_gadget_data(num_qubits)
    assert len(data) == num_leg_bytes(num_qubits) + PHASE_NBYTES
    assert np.all(data == 0)
    assert np.all(data[-PHASE_NBYTES:] == 0)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,legs",
    [
        (num_qubits, rng.integers(0, 4, size=num_qubits, dtype=np.uint8))
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_legs_np(num_qubits: int, legs: PauliArray) -> None:
    data = zero_gadget_data(num_qubits)
    data_legs = get_gadget_legs(data)
    assert len(data_legs) == round_num_legs(num_qubits)
    assert np.all(data_legs == 0)
    set_gadget_legs(data, legs)
    assert all(data[-PHASE_NBYTES:] == 0)
    data_legs = get_gadget_legs(data)
    assert len(data_legs) == round_num_legs(num_qubits)
    assert np.array_equal(data_legs[:num_qubits], legs)
    assert np.all(data_legs[num_qubits:] == 0)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,legs,phase",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.uniform(0, 4 * np.pi),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_phase_np(num_qubits: int, legs: PauliArray, phase: Phase) -> None:
    data = zero_gadget_data(num_qubits)
    data_phase = get_phase(data)
    assert data_phase == 0.0
    assert is_zero_phase(data_phase)
    set_gadget_legs(data, legs)
    data_phase = get_phase(data)
    assert data_phase == 0.0
    set_phase(data, phase)
    data_legs = get_gadget_legs(data)
    assert len(data_legs) == round_num_legs(num_qubits)
    assert np.array_equal(data_legs[:num_qubits], legs)
    assert np.all(data_legs[num_qubits:] == 0)
    data_phase = get_phase(data)
    assert data_phase == phase % (2 * np.pi)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "prec,phase",
    [
        (prec, rng.uniform(0, 2 * np.pi))
        for prec in [0, 2, 4, 8, 16]
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_phase_frac_conversion(prec: int, phase: Phase) -> None:
    frac = Gadget.phase2frac(phase, prec=prec)
    frac_phase = Gadget.frac2phase(frac)
    assert (
        abs((frac_phase - phase) % (2 * np.pi)) < 2 * 2**-prec
        or np.pi - abs((frac_phase - phase) % (2 * np.pi)) < 2 * 2**-prec
    )


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,legs,phase",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.uniform(0, 4 * np.pi),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_assemble_data(num_qubits: int, legs: PauliArray, phase: Phase) -> None:
    data = Gadget.assemble_data(legs, phase)
    assert np.array_equal(get_gadget_legs(data)[:num_qubits], legs)
    assert np.all(get_gadget_legs(data)[num_qubits:] == 0)
    assert get_phase(data) == phase % (2 * np.pi)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,legs,phase",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.uniform(0, 2 * np.pi),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_properties(num_qubits: int, legs: PauliArray, phase: Phase) -> None:
    data = Gadget.assemble_data(legs, phase)
    g = Gadget(data, num_qubits)
    assert np.array_equal(g._data, data)
    assert np.array_equal(g.legs, legs)
    assert g.phase == phase


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,legs,phase",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.uniform(0, 2 * np.pi),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_changes(num_qubits: int, legs: PauliArray, phase: Phase) -> None:
    # First set legs, then set phase:
    rng = np.random.default_rng(RNG_ALT_SEED)
    g = Gadget.random(num_qubits, rng=rng)
    assert g.num_qubits == num_qubits
    prev_phase = g.phase
    g.legs = legs
    assert np.array_equal(g.legs, legs)
    assert g.phase == prev_phase
    g.phase = phase
    assert np.array_equal(g.legs, legs)
    assert g.phase == phase
    # First set phase, then set legs:
    g = Gadget.random(num_qubits, rng=rng)
    assert g.num_qubits == num_qubits
    prev_legs = g.legs
    g.phase = phase
    assert np.array_equal(g.legs, prev_legs)
    assert g.phase == phase
    g.legs = legs
    assert np.array_equal(g.legs, legs)
    assert g.phase == phase


@pytest.mark.parametrize(
    "num_qubits,lhs,rhs",
    [
        (
            num_qubits,
            Gadget.random(num_qubits, rng=rng),
            Gadget.random(num_qubits, rng=rng),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_overlap(num_qubits: int, lhs: Gadget, rhs: Gadget) -> None:
    overlap = sum(
        lhs_leg != 0 and rhs_leg != 0 and lhs_leg != rhs_leg
        for lhs_leg, rhs_leg in zip(lhs.legs, rhs.legs, strict=True)
    )
    assert lhs.overlap(rhs) == overlap


@pytest.mark.parametrize(
    "num_qubits,legs,phase",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.uniform(0, 2 * np.pi),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_equality(num_qubits: int, legs: PauliArray, phase: Phase) -> None:
    lhs = Gadget.from_legs(legs, phase)
    rhs = Gadget.from_legs(legs, phase)
    assert lhs == rhs
    assert lhs == lhs.clone()


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,legs,phase",
    [
        (
            num_qubits,
            rng.integers(0, 4, size=num_qubits, dtype=np.uint8),
            rng.uniform(0, 2 * np.pi),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_gadget_paulistr(num_qubits: int, legs: PauliArray, phase: Phase) -> None:
    paulistr = "".join(PAULI_CHARS[leg] for leg in legs)
    g = Gadget.from_paulistr(paulistr, phase)
    assert np.array_equal(g.legs, legs)
    assert g.leg_paulistr == paulistr
    assert g.phase == phase
    assert g == Gadget.from_legs(legs, phase)
