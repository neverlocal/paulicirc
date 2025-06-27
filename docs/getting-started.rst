===============
Getting Started
===============
PauliCirc is a library for the vectorized creation and manipulation of quantum circuits consisting of Pauli gadgets.

Install
=======

You can install the latest release from `PyPI <https://pypi.org/project/dag-cbor/>`_ as follows:

.. code-block:: console

    $ pip install --upgrade paulicirc


Usage
=====

Some imports common to all sections below.

>>> import numpy as np
>>> from numpy import pi

Pauli Gadgets
-------------

A Pauli gadget (cf. `arXiv:1906.01734 <https://arxiv.org/abs/1906.01734>`_) is a unitary quantum gate performing a many-qubit rotation about a Pauli axis by a given angle, with rotations about the X, Y and Z axes of the Bloch sphere as the single-qubit cases.
Pauli gadgets are also known as Pauli exponentials, or Pauli evolution gates (cf. `qiskit.circuit.library.PauliEvolutionGate <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.PauliEvolutionGate>`_).

Pauli gadgets are the basic ingredient of quantum circuits in the PauliCirc library, and the :class:`~paulicirc.gadgets.Gadget` class provides an interface to create, access and manipulate individual gadget data.

>>> from paulicirc import Gadget

Constructors
^^^^^^^^^^^^

There are various primitive ways to construct gadgets, implemented as class methods.
A "zero gadget" — one corresponding to the identity rotation — can be constructed via the :meth:`Gadget.zero <paulicirc.gadgets.Gadget.zero>` class method, passing the desired number of qubits:

>>> Gadget.zero(10)
<Gadget: __________, 0π>

A random gadget — one where the rotation axis and angle are independently and uniformly sampled — can be constructed via the :meth:`Gadget.random <paulicirc.gadgets.Gadget.random>` class method, passing the desired number of qubits:

>>> Gadget.random(10)
<Gadget: Z_Z_ZXXX_Y, ~17π/16>

Optionally, an integer seed or a Numpy `random generator <https://numpy.org/doc/stable/reference/random/generator.html>` can be passed as the ``rng`` argument, for reproducibility:

>>> Gadget.random(10, rng=0)
<Gadget: XZYYYY_Z_Y, ~21π/256>

The :meth:`Gadget.from_paulistr <paulicirc.gadgets.Gadget.from_paulistr>` class method can be used to create a gadget from a Paulistring and a phase:

>>> Gadget.from_paulistr("XZYYYY_Z_Y", 0.25744424357926954)
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> Gadget.from_paulistr("Z__XY_", 3*pi/4)
<Gadget: Z__XY_, 3π/4>

The :meth:`Gadget.from_sparse_paulistr <paulicirc.gadgets.Gadget.from_sparse_paulistr>` class method can be used to create a gadget from a sparse Paulistring instead, specified by giving a Paulistring, the qubits to which it applies, and the overall number of qubits:

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

The :meth:`Gadget.from_legs <paulicirc.gadgets.Gadget.from_legs>` class method can be used to construct a gadget from such array data instead of a Paulistring.
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
>>> g.legs = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
>>> g
<Gadget: _XZY_XZY_X, π/8>

An independently mutable copy of a gadget can be obtained via the :meth:`Gadget.clone <paulicirc.gadgets.Gadget.clone>` method:

>>> g = Gadget.random(10, rng=0)
>>> g_copy = g.clone()
>>> g == g_copy
True
>>> g is g_copy
False

Unitary Representation
^^^^^^^^^^^^^^^^^^^^^^

The unitary representation of a gadget can be obtained via the :meth:`Gadget.unitary <paulicirc.gadgets.Gadget.unitary>` method:

>>> g = Gadget.from_paulistr("Z", pi/2)
>>> g.unitary().round(3)
array([[ 1.-0.j,  0.+0.j],
       [ 0.+0.j, -0.+1.j]])

The action of a gadget on a statevector can be computed via the :meth:`Gadget.statevec <paulicirc.gadgets.Gadget.statevec>` method:

>>> state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
>>> g.statevec(state)
array([0.5-0.5j, 0.5+0.5j])
>>> g.statevec(state, normalize_phase=True)
array([0.70710678+0.j, 0.+0.70710678j])

Operations
^^^^^^^^^^

The inverse of a gadget is the gadget with same legs and phase negated, and it can be obtained via the :meth:`Gadget.inverse <paulicirc.gadgets.Gadget.inverse>` method:

>>> g = Gadget.random(10, rng=0)
>>> g
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> g.inverse()
<Gadget: XZYYYY_Z_Y, ~491π/256>

The :meth:`Gadget.commutes_with <paulicirc.gadgets.Gadget.commutes_with>` method can be used to check whether a gadget commutes with another gadget:

>>> g = Gadget.from_paulistr("XY_YX", pi/2)
>>> h = Gadget.from_paulistr("ZZX_X", pi/2)
>>> g.commutes_with(h)
True

The overlap between two gadgets is defined to be the number of qubits where (i) both gadgets have a leg different from ``_`` and (ii) the legs of the two gadgets are different.
Whether two gadgets commute depends on whether their overlap is even, and the overlap can be computed via the :meth:`Gadget.overlap <paulicirc.gadgets.Gadget.overlap>` method:

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
The logic to do so is implemented by the :meth:`Gadget.commute_past <paulicirc.gadgets.Gadget.commute_past>` method.
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

For technical details, see the documentation of the :meth:`Gadget.commute_past <paulicirc.gadgets.Gadget.commute_past>` method and the `euler <https://github.com/neverlocal/euler>`_ package.

Approximation
^^^^^^^^^^^^^

The number of bits of precision used when displaying phases is set to 8 by default, resulting in multiples of :math:`\pi/256`:

>>> g = Gadget.random(10, rng=0)
>>> g
<Gadget: XZYYYY_Z_Y, ~21π/256>

A ``~`` character is in front of the phase is used to indicate that the representation is an approximation.
If the ``~`` character is not present, the phase displayed is equal — up to the current relative/absolute tolerances, see below — to the gadget phase:

>>> Gadget.from_paulistr("Z__XY_", 3*pi/4)
<Gadget: Z__XY_, 3π/4>

The display precision can be altered — temporarily or permanently — via the ``display_prec`` option from :obj:`paulicirc.options <paulicirc.utils.options>`:

>>> import paulicirc
>>> g = Gadget.random(10, rng=0)
>>> print(g)
<Gadget: XZYYYY_Z_Y, ~21π/256>
>>> with paulicirc.options(display_prec=16):
...     print(g)
...
<Gadget: XZYYYY_Z_Y, ~2685π/32768>

Gadgets can be compared for approximate equality, with relative and absolute tolerances set by the ``rtol`` and ``atol`` options from :obj:`paulicirc.options <paulicirc.utils.options>` (default values 1e-5 and 1e-8, respectively):

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

The precise logic used for phase comparison is implemented by the :func:`are_same_phase <paulicirc.gadgets.are_same_phase>` function.
See documentation for the `optmanage <https://optmanage.readthedocs.io/en/latest/>` package for specific usage details on the PauliCirc option manager.


Pauli Circuits
--------------

The core data structure for the library is the :class:`Circuit <paulicirc.circuits.Circuit>` class, a memory-efficient implementation of quantum circuits of Pauli gadgets with vectorized operations:

>>> from paulicirc import Circuit

Constructors
^^^^^^^^^^^^

There are various primitive ways to construct circuits, implemented as class methods.
A "zero circuit" — one where all gadgets are zero gadgets — can be constructed via the :meth:`Circuit.zero <paulicirc.circuits.Circuit.zero>` class method, passing the desired number of gadgets and qubits:

>>> Circuit.zero(20, 10)
<Circuit: 20 gadgets, 10 qubits>

A random circuit — one with independently sampled random gadgets — can be constructed via the `Circuit.random <paulicirc.circuits.Circuit.random>` class method, passing the desired number of gadgets and qubits:

>>> Circuit.random(20, 10)
<Circuit: 20 gadgets, 10 qubits>

Optionally, an integer seed or a Numpy `random generator <https://numpy.org/doc/stable/reference/random/generator.html>` can be passed as the ``rng`` argument, for reproducibility:

>>> Circuit.random(20, 10, rng=0)
<Circuit: 20 gadgets, 10 qubits>

A circuit can be constructed from a given list of gadgets via the :meth:`Circuit.from_gadgets <paulicirc.circuits.Circuit.from_gadgets>` class method, passing the desired iterable of gadgets:

>>> Circuit.from_gadgets(
...     Gadget.from_sparse_paulistr("Z", q, 10, pi/2)
...     for q in range(10)
... )
<Circuit: 10 gadgets, 10 qubits>

String Representation
^^^^^^^^^^^^^^^^^^^^^

The string representation of circuits is intentionally opaque, because real-world Pauli circuits quickly get too large to effectively represent.

>>> circ = Circuit.random(20, 10, rng=0)
>>> circ
<Circuit: 20 gadgets, 10 qubits>

The circuit listing object (an instance of :meth:`CircuitListing <paulicirc.circuits.CircuitListing>`) displays an explicit representation of the circuit:

>>> circ.listing
 0 ~351π/256 XXYYZ__ZY_
 1 ~333π/256 Z__ZYYZ_ZY
 2    ~11π/8 XYYX__ZZ_X
 3 ~199π/256 XX_XYXYZ_Z
 4  ~69π/256 XYZXXXZXZZ
 5 ~369π/256 ZXZYX_XXZX
 6 ~269π/256 YXY_ZZ_XYZ
 7 ~159π/256 YZZ_XZ__YZ
 8 ~249π/256 ZY__ZX__ZY
 9 ~455π/256 XZX_ZYYYYX
10 ~239π/128 ZXZX__Z_XY
11 ~183π/256 _ZXYZYZXYX
12 ~293π/256 _YZZX_ZYYY
13 ~165π/256 Z_ZZ_YZ__X
14   ~19π/16 _ZXZXYZYY_
15 ~173π/256 YYXX_YYY__
16 ~201π/256 ZZZ_YZZY_X
17   ~57π/32 Z_XZ_YZZ_Y
18   ~29π/64 Z_ZXZXYXXZ
19 ~319π/256 YYXXYYYZXY

The circuit listing object can be indexed to select individual gadgets within the circuit:

>>> circ.listing[11]
11 ~183π/256 _ZXYZYZXYX

The circuit listing object can also be sliced to select gadget ranges within the circuit:

>>> circ.listing[:8]
0 ~351π/256 XXYYZ__ZY_
1 ~333π/256 Z__ZYYZ_ZY
2    ~11π/8 XYYX__ZZ_X
3 ~199π/256 XX_XYXYZ_Z
4  ~69π/256 XYZXXXZXZZ
5 ~369π/256 ZXZYX_XXZX
6 ~269π/256 YXY_ZZ_XYZ
7 ~159π/256 YZZ_XZ__YZ

Properties
^^^^^^^^^^

The concise string representation of a circuit displays the number of gadgets and number of qubits:

>>> circ = Circuit.random(4, 5, rng=0)
>>> circ
<Circuit: 4 gadgets, 5 qubits>
>>> circ.num_gadgets
4
>>> circ.num_qubits
5

The phase array and leg matrix for the circuit can be accessed in vectorized form:

>>> circ.phases
array([5.73501243, 3.81160499, 4.58356207, 3.41569656])
>>> circ.legs
array([[1, 1, 3, 3, 2],
       [3, 1, 2, 1, 2],
       [2, 2, 2, 1, 0],
       [0, 3, 2, 3, 0]], dtype=uint8)

Circuits are mutable, with the possibility of setting both phase and legs:

>>> circ.phases = [0, pi/2, pi, 3*pi/2]
>>> circ.phases
array([0.        , 1.57079633, 3.14159265, 4.71238898])
>>> circ.legs = [
...     [0, 1, 2, 3, 0],
...     [1, 2, 3, 0, 1],
...     [2, 3, 0, 1, 2],
...     [3, 0, 1, 2, 3]
... ]
>>> circ.legs
array([[0, 1, 2, 3, 0],
       [1, 2, 3, 0, 1],
       [2, 3, 0, 1, 2],
       [3, 0, 1, 2, 3]], dtype=uint8)

An independently mutable copy of a circuit can be obtained via the :meth:`Circuit.clone <paulicirc.circuits.Circuit.clone>` method:

>>> circ = Circuit.random(4, 5, rng=0)
>>> circ_copy = circ.clone()
>>> g == g_copy
True
>>> g is g_copy
False

Unitary Representation
^^^^^^^^^^^^^^^^^^^^^^

The unitary representation of a circuit can be obtained via the :meth:`Circuit.unitary <paulicirc.circuits.Circuit.unitary>` method:

>>> circ = Circuit.from_gadgets([
...     Gadget.from_paulistr("Z", pi/2),
...     Gadget.from_paulistr("X", pi/2),
...     Gadget.from_paulistr("Z", pi/2),
... ])
>>> circ.unitary().round(3)
array([[ 0.707+0.j,  0.707-0.j],
       [ 0.707-0.j, -0.707+0.j]])

The action of a circuit on a statevector can be computed via the :meth:`Circuit.statevec <paulicirc.circuits.Circuit.statevec>` method:

>>> state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
>>> circ.statevec(state).round(3)
array([ 1.+0.j, -0.-0.j])

Operations
^^^^^^^^^^

Circuits behave like sequences of gadgets.

>>> circ = Circuit.random(8, 5, rng=0)
>>> circ
<Circuit: 8 gadgets, 5 qubits>
>>> circ.listing
0 ~209π/128 XXYYZ
1    ~π/256 YXZXZ
2 ~439π/256 ZZZX_
3  ~17π/256 _YZY_
4 ~187π/128 X_YZ_
5  ~45π/128 YZYXZ
6 ~221π/128 XXYYX
7 ~277π/256 _ZZYZ

The length of the circuit is the number of gadgets, which can be iterated over:

>>> len(circ)
8
>>> for gadget in circ:
...     print(gadget)
...
<Gadget: XXYYZ, ~209π/128>
<Gadget: YXZXZ, ~π/256>
<Gadget: ZZZX_, ~439π/256>
<Gadget: _YZY_, ~17π/256>
<Gadget: X_YZ_, ~187π/128>
<Gadget: YZYXZ, ~45π/128>
<Gadget: XXYYX, ~221π/128>
<Gadget: _ZZYZ, ~277π/256>

Individual gadgets can be accessed by indexing:

>>> circ[2]
<Gadget: ZZZX_, ~187π/128>

Sub-circuits can be accessed by slicing:

>>> circ[:4]
<Circuit: 4 gadgets, 5 qubits>
>>> circ[:4].listing
0 ~209π/128 XXYYZ
1    ~π/256 YXZXZ
2 ~439π/256 ZZZX_
3  ~17π/256 _YZY_

Slices can have non-trivial step:

>>> circ[::3]
<Circuit: 3 gadgets, 5 qubits>
>>> circ[::3].listing
0 ~209π/128 XXYYZ
1  ~17π/256 _YZY_
2 ~221π/128 XXYYX

Sub-circuits with irregular step can be accessed by specifying multiple indices:

>>> circ[[0, 2, 6]]
<Circuit: 3 gadgets, 5 qubits>
>>> circ[[0, 2, 6]].listing
0 ~209π/128 XXYYZ
1 ~439π/256 ZZZX_
2 ~221π/128 XXYYX

Circuits are mutable, with the possibility of setting individual gadgets or sub-circuits:

>>> circ[0] = Gadget.from_paulistr("XYZXY", pi/2)
>>> circ[0]
<Gadget: XYZXY, π/2>
>>> circ[::2] = circ[1::2]
>>> circ.listing
0       π/2 XYZXY
1       π/2 XYZXY
2  ~17π/256 _YZY_
3  ~17π/256 _YZY_
4  ~45π/128 YZYXZ
5  ~45π/128 YZYXZ
6 ~277π/256 _ZZYZ
7 ~277π/256 _ZZYZ

The inverse of a circuit is the circuit with same legs and phase negated, and it can be obtained via the :meth:`Circuit.inverse <paulicirc.circuits.Circuit.inverse>` method:

>>> circ = Circuit.random(4, 5, rng=0)
>>> circ.listing
0 ~467π/256 XXYYZ
1 ~311π/256 YXZXZ
2 ~187π/128 ZZZX_
3 ~139π/128 _YZY_
>>> circ.inverse()
<Circuit: 4 gadgets, 5 qubits>
>>> circ.inverse().listing
0 ~117π/128 _YZY_
1  ~69π/128 ZZZX_
2 ~201π/256 YXZXZ
3  ~45π/256 XXYYZ
