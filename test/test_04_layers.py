from typing import Any
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
import pytest

from paulicirc.builders import CircuitBuilder
from paulicirc.gadgets import Gadget
from paulicirc.layers import Layer, LayeredCircuit
from paulicirc.utils.numpy import canonicalize_phase
from paulicirc.layers import Layer, LayeredCircuit
from paulicirc.circuits import Circuit

RNG_SEED = 0
RNG_ALT_SEED = 1
NUM_RNG_SAMPLES = 10
NUM_QUBITS_RANGE = range(0, 8)

rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits",
    [num_qubits for num_qubits in NUM_QUBITS_RANGE],
)
def test_empty_layered_circuit(num_qubits: int) -> None:
    layered_circ = LayeredCircuit(num_qubits)
    assert layered_circ.num_qubits == num_qubits
    assert layered_circ.num_layers == 0
    assert not layered_circ.layers
    assert not list(layered_circ)
    assert len(layered_circ) == 0
    circ = layered_circ.circuit()
    assert circ.num_qubits == num_qubits
    assert circ.num_gadgets == 0


SINGLE_GATE_LAYERS_TEST_CASES: list[
    tuple[int, str, tuple[Any, ...], dict[str, Any], list[list[tuple[str, float]]]]
] = [
    *[
        (1, "rz", (t, 0), {}, [[("Z", t)]])
        for t in [0.0, pi, pi / 2, -pi / 2, pi / 4, -pi / 4]
    ],
    *[
        (1, "rx", (t, 0), {}, [[("X", t)]])
        for t in [0.0, pi, pi / 2, -pi / 2, pi / 4, -pi / 4]
    ],
    *[
        (1, "ry", (t, 0), {}, [[("Y", t)]])
        for t in [0.0, pi, pi / 2, -pi / 2, pi / 4, -pi / 4]
    ],
    *[
        (
            1,
            "h",
            (0,),
            {"xzx": xzx},
            (
                [
                    [("X", pi / 2)],
                    [("Z", pi / 2)],
                    [("X", pi / 2)],
                ]
                if xzx
                else [
                    [("Z", pi / 2)],
                    [("X", pi / 2)],
                    [("Z", pi / 2)],
                ]
            ),
        )
        for xzx in (False, True)
    ],
    *[
        (
            1,
            "hdg",
            (0,),
            {"xzx": xzx},
            (
                [
                    [("X", -pi / 2)],
                    [("Z", -pi / 2)],
                    [("X", -pi / 2)],
                ]
                if xzx
                else [
                    [("Z", -pi / 2)],
                    [("X", -pi / 2)],
                    [("Z", -pi / 2)],
                ]
            ),
        )
        for xzx in (False, True)
    ],
    (2, "cz", (0, 1), {}, [[("Z_", -pi / 2), ("_Z", -pi / 2), ("ZZ", pi / 2)]]),
    (
        2,
        "cx",
        (0, 1),
        {},
        [
            [("Z_", -pi / 2), ("_Z", pi / 2)],
            [("_X", pi / 2)],
            [("_Z", -pi / 2), ("ZZ", pi / 2)],
            [("_X", -pi / 2)],
            [("_Z", -pi / 2)],
        ],
    ),
    (
        2,
        "cx",
        (1, 0),
        {},
        [
            [("_Z", -pi / 2), ("Z_", pi / 2)],
            [("X_", pi / 2)],
            [("Z_", -pi / 2), ("ZZ", pi / 2)],
            [("X_", -pi / 2)],
            [("Z_", -pi / 2)],
        ],
    ),
    (
        2,
        "cy",
        (0, 1),
        {},
        [
            [("Z_", -pi / 2), ("_X", pi / 2)],
            [("_Z", -pi / 2), ("ZZ", pi / 2)],
            [("_X", -pi / 2)],
        ],
    ),
    (
        2,
        "cy",
        (1, 0),
        {},
        [
            [("_Z", -pi / 2), ("X_", pi / 2)],
            [("Z_", -pi / 2), ("ZZ", pi / 2)],
            [("X_", -pi / 2)],
        ],
    ),
    (
        2,
        "swap",
        (0, 1),
        {},
        [
            [("_Z", pi / 2)],
            [("_X", pi / 2)],
            [("_Z", -pi / 2), ("ZZ", pi / 2)],
            [("X_", pi / 2), ("_X", -pi / 2)],
            [("_Z", -pi / 2), ("Z_", -pi / 2), ("ZZ", pi / 2)],
            [("X_", -pi / 2), ("_X", pi / 2)],
            [("_Z", -pi / 2), ("Z_", pi), ("ZZ", pi / 2)],
            [("_X", -pi / 2)],
            [("_Z", -pi / 2)],
        ],
    ),
]


@pytest.mark.parametrize(
    "num_qubits,gate,args,kwargs,layer_gadgets_list", SINGLE_GATE_LAYERS_TEST_CASES
)
def test_single_gate_layer(
    num_qubits: int,
    gate: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    layer_gadgets_list: list[list[tuple[str, float]]],
) -> None:
    layers = [
        Layer.from_gadgets(
            (Gadget.from_paulistr(paulis, phase) for paulis, phase in layer_gadgets),
            num_qubits=num_qubits,
        )
        for layer_gadgets in layer_gadgets_list
    ]
    layered_circ = LayeredCircuit(num_qubits)
    builder = CircuitBuilder(num_qubits)
    getattr(builder, gate)(*args, **kwargs)
    layered_circ.extend(builder)
    assert layered_circ.num_qubits == num_qubits
    assert layered_circ.num_layers == len(layers)
    for idx, (layer, layered_circ_layer) in enumerate(zip(layers, layered_circ._layers)):
        assert layer == layered_circ_layer, (
            idx,
            layer,
            list(layer),
            layered_circ_layer,
            list(layered_circ_layer),
        )


SINGLE_GATE_UNITARY_TEST_CASES: list[
    tuple[
        int,  # num_qubits
        str,  # gate label
        tuple[Any, ...],  # gate args
        dict[str, Any],  # gate kwargs
        np.ndarray[Any, np.dtype[Any]],  # expected unitary
    ]
] = [
    *[
        (1, "rz", (t, 0), {}, np.array([[1, 0], [0, exp(t * 1j)]]))
        for t in [0.0, pi, pi / 2, -pi / 2, pi / 4, -pi / 4]
    ],
    *[
        (
            1,
            "rx",
            (t, 0),
            {},
            np.array([[cos(t / 2), -1j * sin(t / 2)], [-1j * sin(t / 2), cos(t / 2)]]),
        )
        for t in [0.0, pi, pi / 2, -pi / 2, pi / 4, -pi / 4]
    ],
    *[
        (
            1,
            "ry",
            (t, 0),
            {},
            np.array([[cos(t / 2), -sin(t / 2)], [sin(t / 2), cos(t / 2)]]),
        )
        for t in [0.0, pi, pi / 2, -pi / 2, pi / 4, -pi / 4]
    ],
    *[
        (
            1,
            "h",
            (0,),
            {"xzx": xzx},
            np.array([[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]),
        )
        for xzx in (False, True)
    ],
    *[
        (
            1,
            "hdg",
            (0,),
            {"xzx": xzx},
            np.array([[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]),
        )
        for xzx in (False, True)
    ],
    (
        2,
        "cz",
        (0, 1),
        {},
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ]
        ),
    ),
    (
        2,
        "cx",
        (0, 1),
        {},
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        ),
    ),
    (
        2,
        "cx",
        (1, 0),
        {},
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
            ]
        ),
    ),
    (
        2,
        "cy",
        (0, 1),
        {},
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ]
        ),
    ),
    (
        2,
        "swap",
        (0, 1),
        {},
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
    ),
    (
        3,
        "ccz",
        (0, 1, 2),
        {},
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ]
        ),
    ),
    (
        3,
        "ccx",
        (0, 1, 2),
        {},
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        ),
    ),
    (
        3,
        "ccx",
        (0, 2, 1),
        {},
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        ),
    ),
    (
        3,
        "ccy",
        (0, 1, 2),
        {},
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 0, 0, 1j, 0],
            ]
        ),
    ),
    (
        3,
        "cswap",
        (0, 1, 2),
        {},
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "num_qubits,gate,args,kwargs,unitary", SINGLE_GATE_UNITARY_TEST_CASES
)
def test_single_gate_unitary_layered(
    num_qubits: int,
    gate: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    unitary: np.ndarray[Any, np.dtype[Any]],
) -> None:
    unitary = unitary.astype(np.complex128)
    canonicalize_phase(unitary)
    layered_circ = LayeredCircuit(num_qubits)
    builder = CircuitBuilder(num_qubits)
    getattr(builder, gate)(*args, **kwargs)
    layered_circ.extend(builder)
    assert layered_circ.num_qubits == num_qubits
    u = layered_circ.unitary()
    circ = layered_circ.circuit()
    circ_u = circ.unitary()
    assert u.shape == unitary.shape
    assert np.allclose(u, unitary)
    assert circ_u.shape == unitary.shape
    assert np.allclose(circ_u, unitary)


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES // 2)
        for _ in range(NUM_RNG_SAMPLES // 2)
    ],
)
def test_building_from_circuit_append(
    num_qubits: int, num_gadgets: int, seed: int
) -> None:
    layered_circ = LayeredCircuit(num_qubits)
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    for g in circ:
        layered_circ.append(g)
    assert np.allclose(layered_circ.unitary(), circ.unitary())


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES // 2)
        for _ in range(NUM_RNG_SAMPLES // 2)
    ],
)
def test_building_from_circuit_extend_circ(
    num_qubits: int, num_gadgets: int, seed: int
) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    layered_circ = LayeredCircuit(num_qubits)
    layered_circ.extend(circ)
    assert np.allclose(layered_circ.unitary(), circ.unitary())


rng = np.random.default_rng(RNG_SEED)


@pytest.mark.parametrize(
    "num_qubits,num_gadgets,seed",
    [
        (num_qubits, num_gadgets, rng.integers(0, 65536))
        for num_qubits in NUM_QUBITS_RANGE
        for num_gadgets in rng.integers(0, 20, size=NUM_RNG_SAMPLES // 2)
        for _ in range(NUM_RNG_SAMPLES // 2)
    ],
)
def test_building_from_circuit_extend_list(
    num_qubits: int, num_gadgets: int, seed: int
) -> None:
    circ = Circuit.random(num_gadgets, num_qubits, rng=seed)
    layered_circ = LayeredCircuit(num_qubits)
    layered_circ.extend(list(circ))
    assert np.allclose(layered_circ.unitary(), circ.unitary())
