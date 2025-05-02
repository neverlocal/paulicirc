import numpy as np
import pytest

from pauliobf._numpy import normalise_phase
from pauliobf.circuits import Circuit
from pauliobf.gadgets import Gadget, are_same_phase

RNG_SEED = 0
RNG_ALT_SEED = 1
NUM_RNG_SAMPLES = 10
NUM_QUBITS_RANGE = range(0, 9)

rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets",
    [
        (num_qubits, num_gadgets)
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_zero_circuit(num_qubits: int, num_gadgets: int) -> None:
    circ = Circuit.zero(num_gadgets, num_qubits)
    assert circ.num_qubits == num_qubits
    assert circ.num_gadgets == num_gadgets
    gs = list(circ)
    assert len(gs) == num_gadgets
    assert all(g.num_qubits == num_qubits for g in gs)
    assert all(g.is_zero for g in gs)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_random_circuit(num_qubits: int, num_gadgets: int, seed: int) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    assert circ.num_qubits == num_qubits
    assert circ.num_gadgets == num_gadgets
    gs = list(circ)
    assert len(gs) == num_gadgets
    assert all(g.num_qubits == num_qubits for g in gs)
    assert all(are_same_phase(g.phase, phase) for g, phase in zip(gs, circ.phases))
    v = np.eye(2**num_qubits, dtype=np.complex128)
    for i, g in enumerate(gs):
        v = g.unitary() @ v
    normalise_phase(v)
    u = circ.unitary()
    assert u.dtype == v.dtype
    assert u.shape == v.shape
    assert np.allclose(u, v)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_circuit_from_gadgets(num_qubits: int, num_gadgets: int, seed: int) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    assert circ == Circuit.from_gadgets(circ, num_qubits)
    if num_gadgets > 0:
        assert circ == Circuit.from_gadgets(circ, None)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_circuit_clone(num_qubits: int, num_gadgets: int, seed: int) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    assert circ == circ.clone()


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed,fast",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536), fast)
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
        for fast in (False, True)
    ],
)
def test_circuit_iter_gadgets(
    num_qubits: int, num_gadgets: int, seed: int, fast: bool
) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    gs = list(circ)
    assert all(g == h for g, h in zip(gs, circ.iter_gadgets(fast=fast), strict=True))


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_circuit_getitem(
    num_qubits: int,
    num_gadgets: int,
    seed: int,
) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    gs = list(circ)
    assert all(g == circ[i] for i, g in enumerate(gs))


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed,start,stop,step",
    [
        (
            num_qubits,
            num_gadgets,
            rng.integers(0, 65536),
            (start := int(rng.integers(0, num_gadgets))),
            (stop := int(rng.integers(start, num_gadgets + 1))),
            (step := int(rng.integers(1, num_gadgets + 1))),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(1, 20, size=NUM_RNG_SAMPLES)
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_circuit_getitem_slice(
    num_qubits: int,
    num_gadgets: int,
    seed: int,
    start: int,
    stop: int,
    step: int,
) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    gs = list(circ)
    assert all(g == circ[i] for i, g in enumerate(gs))
    slice_idxs = range(num_gadgets)[s := slice(start, stop, step)]
    slice_gs = circ[s]
    assert all(gs[i] == g for i, g in zip(slice_idxs, slice_gs, strict=True))


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(1, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_circuit_setitem(
    num_qubits: int,
    num_gadgets: int,
    seed: int,
) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    original_gs = list(circ)
    new_gs = [Gadget.random(num_qubits, rng=seed + 1 + i) for i in range(num_gadgets)]
    for i, g in enumerate(new_gs):
        circ[i] = g
        gs_i = list(circ)
        assert gs_i[: i + 1] == new_gs[: i + 1]
        assert gs_i[i + 1 :] == original_gs[i + 1 :]
    assert list(circ) == new_gs


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed,start,stop,step",
    [
        (
            num_qubits,
            num_gadgets,
            rng.integers(0, 65536),
            (start := int(rng.integers(0, num_gadgets))),
            (stop := int(rng.integers(start, num_gadgets + 1))),
            (step := int(rng.integers(1, num_gadgets + 1))),
        )
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(1, 20, size=NUM_RNG_SAMPLES)
        for _ in range(NUM_RNG_SAMPLES)
    ],
)
def test_circuit_setitem_slice(
    num_qubits: int,
    num_gadgets: int,
    seed: int,
    start: int,
    stop: int,
    step: int,
) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    original_gs = list(circ)
    new_gs = [Gadget.random(num_qubits, rng=seed + 1 + i) for i in range(num_gadgets)]
    slice_idxs = range(num_gadgets)[s := slice(start, stop, step)]
    slice_gs = [new_gs[i] for i in slice_idxs]
    slice_circ = Circuit.from_gadgets(slice_gs, num_qubits)
    circ[s] = slice_circ
    slice_idxs_set = set(slice_idxs)
    for i, g in enumerate(circ):
        if i in slice_idxs_set:
            assert g == new_gs[i]
        else:
            assert g == original_gs[i]


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_circuit_inverse(num_qubits: int, num_gadgets: int, seed: int) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    gs = list(circ)
    circ_inv = circ.inverse()
    gs_inv = list(circ_inv)
    assert all(g.inverse() == g_inv for g, g_inv in zip(gs, reversed(gs_inv)))


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_pairs,seed",
    [
        (num_qubits, num_pairs, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_pairs in rng.integers(0, 20, size=NUM_RNG_SAMPLES)
    ],
)
def test_circuit_inverse_pairs(num_qubits: int, num_pairs: int, seed: int) -> None:
    circ = Circuit.random_inverse_pairs(num_pairs, num_qubits, rng=seed)
    assert circ.num_qubits == num_qubits
    assert circ.num_gadgets == 2 * num_pairs
    assert all(
        circ[2 * pair_idx].inverse() == circ[2 * pair_idx + 1]
        for pair_idx in range(num_pairs)
    )
