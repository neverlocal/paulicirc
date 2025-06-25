=========
PauliCirc
=========

.. image:: https://img.shields.io/badge/python-3.10+-green.svg
    :target: https://docs.python.org/3.10/
    :alt: Python versions

.. image:: https://img.shields.io/pypi/v/paulicirc.svg
    :target: https://pypi.python.org/pypi/paulicirc/
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/status/paulicirc.svg
    :target: https://pypi.python.org/pypi/paulicirc/
    :alt: PyPI status

.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: https://github.com/python/mypy
    :alt: Checked with Mypy

.. image:: https://readthedocs.org/projects/paulicirc/badge/?version=latest
    :target: https://paulicirc.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/neverlocal/paulicirc/actions/workflows/python-pytest.yml/badge.svg
    :target: https://github.com/neverlocal/paulicirc/actions/workflows/python-pytest.yml
    :alt: Python package status


.. contents::


PauliCirc is a library for the vectorized creation and manipulation of quantum circuits consisting of Pauli gadgets.

Install
=======

You can install the latest release from `PyPI <https://pypi.org/project/dag-cbor/>`_ as follows:

.. code-block:: console

    $ pip install --upgrade paulicirc


Usage
=======

This library depends on

Pauli Gadgets
-------------

A Pauli gadget (cf. `arXiv:1906.01734 <https://arxiv.org/abs/1906.01734>`_) is a unitary quantum gate performing a many-qubit rotation about a Pauli axis by a given angle, with rotations about the X, Y and Z axes of the Bloch sphere as the single-qubit cases.
Pauli gadgets are also known as Pauli exponentials, or Pauli evolution gates (cf. `qiskit.circuit.library.PauliEvolutionGate <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.PauliEvolutionGate>`_).

Pauli gadgets are the basic ingredient of quantum circuits in the PauliCirc library, and the `Gadget <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#gadget>`_ class provides an interface to create, access and manipulate individual gadget data.

>>> from paulicirc import Gadget

Constructors
^^^^^^^^^^^^

There are various primitive ways to construct gadgets, implemented as class methods.
A "zero gadget" — one corresponding to the identity rotation — can be constructed via the `Gadget.zero <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.zero>`_ class method, passing the desired number of qubits:

>>> Gadget.zero(10)
<Gadget: __________, 0π>

A random gadget — one where the rotation axis and angle are independently and uniformly sampled — can be constructed via the `Gadget.random <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.random>`_ class method, passing the desired number of qubits:

>>> Gadget.random(10)
<Gadget: Z_Z_ZXXX_Y, ~17π/16>

Optionally, an integer seed or a Numpy `random generator <https://numpy.org/doc/stable/reference/random/generator.html>` can be passed as the ``rng`` argument, for reproducibility:

>>> Gadget.random(10, rng=0)
<Gadget: XZYYYY_Z_Y, ~21π/256>

The `Gadget.from_paulistr <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.from_paulistr>`_ class method can be used to create a gadget from a Paulistring and a phase:

>>> Gadget.from_paulistr("XZYYYY_Z_Y", 0.25744424357926954)
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> Gadget.from_paulistr("Z__XY_", 3*pi/4)
<Gadget: Z__XY_, 3π/4>

The `Gadget.from_sparse_paulistr <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.from_sparse_paulistr>`_ class method can be used to create a gadget from a sparse Paulistring instead, specified by giving a Paulistring, the qubits to which it applies, and the overall number of qubits:

>>> Gadget.from_sparse_paulistr("ZXY", [0, 3, 4], 6, 3*pi/4)
<Gadget: Z__XY_, 3π/4>


Properties
^^^^^^^^^^

The rotation axis is known as the gadget's legs. The legs are a Paulistring, i.e. a string of ``_``, ``X``, ``Y`` or ``Z`` characters indicating the axis component along each qubit, where ``_`` indicates no rotation action on the corresponding qubit:

>>> g = Gadget.random(10, rng=0)
>>> g.leg_paulistr
'XZYYYY_Z_Y'

The number of legs coincides with the number of qubits upon which the gadget is defined:

>>> g.num_qubits
10

At a lower level, the legs are instead represented as an array of integers 0-4:

>>> g.legs
array([1, 2, 3, 3, 3, 3, 0, 2, 0, 3], dtype=uint8)

The `Gadget.from_legs <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.from_legs>`_ class method can be used to construct a gadget from such array data instead of a Paulistring.
The rotation angle is known as the gadget's phase, represented as a floating point number:

>>> g.phase
0.25744424357926954

Approximate representations of the gadget's phase as a fraction of :math:`\pi` are also available:

>>> g.phase_frac
Fraction(21, 256)
>>> g.phase_str
'~21π/256'

Gadgets are mutable, with the possibility of setting both phase and legs:

>>> g = Gadget.random(10, rng=0)
>>> g
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> g.phase = pi/8
>>> g
<Gadget: XZYYYY_Z_Y, π/8>
>>> g.legs = "XYZ__ZYX__"
>>> g
<Gadget: XYZ__ZYX__, π/8>
>>> new_legs = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.uint8)
>>> g.legs = new_legs
>>> g
<Gadget: _XZY_XZY_X, π/8>

An independently mutable copy of a gadget can be obtained via the `Gadget.clone <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.clone>`_ method:

>>> g = Gadget.random(10, rng=0)
>>> g_copy = g.clone()
>>> g == g_copy
True
>>> g is g_copy
False

Unitary Representation
^^^^^^^^^^^^^^^^^^^^^^

The unitary representation of a gadget can be obtained via the `Gadget.unitary <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.unitary>`_ method:

>>> g = Gadget.from_paulistr("Z", pi/2)
>>> g.unitary().round(3)
array([[ 1.-0.j,  0.+0.j],
       [ 0.+0.j, -0.+1.j]])

The action of a gadget on a statevector can be computed via the `Gadget.statevec <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.statevec>`_ method:

>>> state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
>>> g.statevec(state)
array([0.5-0.5j, 0.5+0.5j])
>>> g.statevec(state, normalize_phase=True)
array([0.70710678+0.j, 0.+0.70710678j])

Operations
^^^^^^^^^^

The inverse of a gadget is the gadget with same legs and phase negated, and it can be obtained via the `Gadget.inverse <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.inverse>`_ method:

>>> g = Gadget.random(10, rng=0)
>>> g
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> g.inverse()
<Gadget: XZYYYY_Z_Y, ~491π/256>

The `Gadget.commutes_with <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.commutes_with>`_ method can be used to check whether a gadget commutes with another gadget:

>>> g = Gadget.from_paulistr("XY_YX", pi/2)
>>> h = Gadget.from_paulistr("ZZX_X", pi/2)
>>> g.commutes_with(h)
True

The overlap between two gadgets is defined to be the number of qubits where (i) both gadgets have a leg different from ``_`` and (ii) the legs of the two gadgets are different.
Whether two gadgets commute depends on whether their overlap is even, and the overlap can be computed via the `Gadget.overlap <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.overlap>`_ method:

>>> g.overlap(h)
2

As an example of gadgets which don't commute:

>>> g = Gadget.from_paulistr("XY", pi/2)
>>> h = Gadget.from_paulistr("_Z", -pi/4)
>>> g.commutes_with(h)
False
>>> g.overlap(h)
1

Gadgets which don't commute can still be "commuted past" each other by changing their phases and introducing a third gadget with a specially chosen phase.
The logic to do so is implemented by the `Gadget.commute_past <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.commute_past>`_ method.
As its second argument, the method takes a numeric code 0-7.
Code 0 means to not commute the gadgets:

>>> g.commute_past(h, 0)
(<Gadget: XY, π/2>, <Gadget: _Z, 7π/4>, <Gadget: __, 0π>)

Codes 1-7 correspond to six possible ways to commute the gadgets past each other, according to `Euler angle conversions <https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix>`_:

>>> g.commute_past(h, 1)
(<Gadget: _Z, 3π/2>, <Gadget: XX, π/2>, <Gadget: _Z, ~π/4>)
>>> g.commute_past(h, 2)
(<Gadget: XX, ~3π/4>, <Gadget: _Z, π/2>, <Gadget: XX, 3π/2>)
>>> g.commute_past(h, 3)
(<Gadget: XY, ~0π>, <Gadget: XX, ~π/4>, <Gadget: XY, π/2>)
>>> g.commute_past(h, 4)
(<Gadget: XX, ~π/4>, <Gadget: XY, π/2>, <Gadget: XX, ~0π>)
>>> g.commute_past(h, 5)
(<Gadget: _Z, 3π/2>, <Gadget: XY, ~π/4>, <Gadget: XX, π/2>)
>>> g.commute_past(h, 6)
(<Gadget: _Z, ~0π>, <Gadget: XX, ~π/4>, <Gadget: XY, π/2>)
>>> g.commute_past(h, 7)
(<Gadget: XX, ~π/4>, <Gadget: _Z, ~0π>, <Gadget: XY, π/2>)

For technical details, see the documentation of the `Gadget.commute_past <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.commute_past>`_ method and the `euler <https://github.com/neverlocal/euler>`_ package.

Approximation
^^^^^^^^^^^^^

The number of bits of precision used when displaying phases is set to 8 by default, resulting in multiples of :math:`\pi/256`.
The precision can be altered — temporarily or permanently — via the ``display_prec`` option from `paulicirc.options <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.utils.html#paulicircoptions>`_:

>>> import paulicirc
>>> with paulicirc.options(display_prec=16):
...     print(g.phase_str)
...
~2685π/32768

Gadgets can be compared for approximate equality, with relative and absolute tolerances set by the ``rtol`` and ``atol`` options from `paulicirc.options <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.utils.html#paulicircoptions>`_ (default values 1e-5 and 1e-8, respectively):

>>> g = Gadget.random(10, rng=0)
>>> g
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> g.phase
0.25744424357926954
>>> g == Gadget.from_paulistr("XZYYYY_Z_Y", 0.25744424357926954)
True
>>> g == Gadget.from_paulistr("XZYYYY_Z_Y", 0.257442)
True
>>> g == Gadget.from_paulistr("XZYYYY_Z_Y", 0.25744)
False

Note that the precision used by equality comparison is usually much higher than the display precision, so that gadgets which test as not approximately equal may be printed as having the same phase:

>>> g = Gadget.random(10, rng=0)
>>> g
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> Gadget.from_paulistr("XZYYYY_Z_Y", 0.25744)
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> g.phase
0.25744424357926954

The precise logic used for phase comparison is implemented by the `are_same_phase <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.are_same_phase>` function.
See documentation for the `optmanage <https://optmanage.readthedocs.io/en/latest/>` package for specific usage details on the PauliCirc option manager.


Pauli Circuits
--------------

The core data structure for the library is the `Circuit <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.circuits.html#circuit>`_ class, a memory-efficient implementation of quantum circuits of Pauli gadgets with vectorized operations:

>>> from paulicirc import Circuit

Constructors
^^^^^^^^^^^^
There are various primitive ways to construct circuits, implemented as class methods.
A "zero circuit" — one where all gadgets are zero gadgets — can be constructed via the `Gadget.zero <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.circuits.html#paulicirc.circuits.Circuit.zero>`_ class method, passing the desired number of qubits:

>>> Gadget.zero(10)
<Gadget: __________, 0π>

A random gadget — one where the rotation axis and angle are independently and uniformly sampled — can be constructed via the `Gadget.random <https://paulicirc.readthedocs.io/en/latest/api/paulicirc.gadgets.html#paulicirc.gadgets.Gadget.random>`_ class method, passing the desired number of qubits:

>>> Gadget.random(10)
<Gadget: Z_Z_ZXXX_Y, ~17π/16>

Optionally, an integer seed or a Numpy `random generator <https://numpy.org/doc/stable/reference/random/generator.html>` can be passed as the ``rng`` argument, for reproducibility:

>>> Gadget.random(10, rng=0)
<Gadget: XZYYYY_Z_Y, ~21π/256>


API
===

For the full API documentation, see https://paulicirc.readthedocs.io/


License
=======

`LGPLv3 © NeverLocal. <LICENSE>`_