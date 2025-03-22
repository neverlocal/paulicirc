"""Spider graphs."""

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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Self,
    TypeAlias,
    TypedDict,
    final,
)
import numpy as np

if TYPE_CHECKING:
    try:
        from networkx import DiGraph  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        DiGraph = Any

if __debug__:
    from typing_validation import validate
    import autoray  # type: ignore[import-untyped]

Matrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[Any]]]
"""
Type alias for complex matrices in a spider graph.
The type specifies that these are NumPy arrays, but in reality any class
which supports :func:`autoray.shape` (e.g. CuPy arrays) can be used here.
"""

Dim: TypeAlias = int
"""Type alias for dimensions."""

DimArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.unsignedinteger[Any]]]
"""Type alias for arrays of dimensions."""

SpiderIdx: TypeAlias = int
"""Type alias for spider indices."""

EdgeData: TypeAlias = tuple[Matrix, SpiderIdx, SpiderIdx]
"""Type alias for edge data, a triple ``(matrix, tail, head)``."""

EdgeIdx: TypeAlias = int
"""Type alias for edge indices."""

EdgeArray: TypeAlias = np.ndarray[
    tuple[int, Literal[2]], np.dtype[np.unsignedinteger[Any]]
]
"""Type alias for arrays of edge arrays, 1D arrays of ``(tail, head)`` pairs."""


def _get_min_dtype(value: int) -> type[np.unsignedinteger[Any]]:
    """Returns the minimum unsigned integer dtype capable of storing the given value."""
    assert value >= 0, "Value must be non-negative."
    if value < 256:
        return np.uint8
    if value < 65536:
        return np.uint16
    if value < 4294967296:
        return np.uint32
    assert value < 18446744073709551616, "Value exceeds capacity of uint64 dtype."
    return np.uint64


_UINT_DTYPE_MAX: Final[Mapping[type[np.unsignedinteger[Any]], int]] = {
    np.uint8: 255,
    np.uint16: 65535,
    np.uint32: 4294967295,
    np.uint64: 18446744073709551615,
}
"""Mapping of available unsigned integer dtypes to the corresponding max value."""


class ContractionArgs(TypedDict, total=True):
    """
    Tensor contraction arguments, passed e.g. to :func:`cotengra.array_contract_tree`.
    """

    inputs: EdgeArray
    """
    The list of wire/spider indices for the legs of each tensor in the tensor network.
    Because all tensors are matrices, this can be returned as an array with 2 columns.
    """

    outputs: Sequence[SpiderIdx]
    """The list of wire/spider indices to be kept as open legs for the contraction."""

    size_dict: Mapping[SpiderIdx, Dim]
    """Mapping of spider indices to their corresponding dimension."""


@final
class SpiderGraph:
    """
    A spider graph is a data structure representing a quiver where nodes correspond to
    phaseless computational basis spiders and edges correspond to arbitrary complex
    matrices between them.
    """

    _matrices: list[Matrix]
    _edges: EdgeArray
    _spider_dims: DimArray
    _num_edges: int
    _num_spiders: int
    _edges_dtype_max: int
    _spiders_dtype_max: int
    _max_dim: int

    def __new__(
        cls,
        *,
        edge_capacity: int | None = None,
        spider_capacity: int | None = None,
    ) -> Self:
        """
        Constructs an empty spider graph.

        Initial capacities for edges and/or spiders can be optionally specified.

        :meta public:
        """
        self = super().__new__(cls)
        self._matrices = []
        self._edges = np.zeros((256, 2), dtype=np.uint8)
        self._spider_dims = np.zeros(256, dtype=np.uint8)
        self._num_edges = 0
        self._num_spiders = 0
        self._max_dim = 1
        self._edges_dtype_max = _UINT_DTYPE_MAX[np.uint8]
        self._spiders_dtype_max = _UINT_DTYPE_MAX[np.uint8]
        if edge_capacity is not None or spider_capacity is not None:
            self.resize_capacity(edge_capacity, spider_capacity)
        return self

    @property
    def matrices(self) -> Sequence[Matrix]:
        """The matrices associated to the edges of this spider graph."""
        return tuple(self._matrices)

    @property
    def edges(self) -> EdgeArray:
        """Edges of this spider graph."""
        view = self._edges.view()
        view.setflags(write=False)
        return view[: self._num_edges].view()

    @property
    def num_edges(self) -> int:
        """Number of edges in this spider graph."""
        return self._num_edges

    @property
    def spiders(self) -> Sequence[SpiderIdx]:
        """The sequence of indices for spiders in this spider graph."""
        return range(self._num_spiders)

    @property
    def spider_dims(self) -> DimArray:
        """Dimensions for spiders in this spider graph."""
        view = self._spider_dims.view()
        view.setflags(write=False)
        return view[: self._num_spiders].view()

    @property
    def num_spiders(self) -> int:
        """Number of spiders in this spider graph."""
        return self._num_spiders

    def add_spiders(self, dims: Iterable[Dim]) -> Sequence[SpiderIdx]:
        """
        Adds spiders with the given dimensions to the spider graph.
        Returns the indices of the new spiders.
        """
        dims = tuple(dims)
        if not dims:
            return ()
        assert self.__validate_spider_dims(dims)
        spider_dims, n = self._spider_dims, self._num_spiders
        ndims = len(dims)
        if (max_dim := max(dims)) > self._spiders_dtype_max:
            t = _get_min_dtype(max_dim)
            self._spider_dims = spider_dims = spider_dims.astype(t)
            self._spiders_dtype_max = _UINT_DTYPE_MAX[t]
        if (new_n := n + ndims) - 1 > self._edges_dtype_max:
            t = _get_min_dtype(new_n - 1)
            self._edges = self._edges.astype(t)
            self._edges_dtype_max = _UINT_DTYPE_MAX[t]
        if new_n > len(spider_dims):
            spider_dims.resize(2 * len(spider_dims), refcheck=False)
        spider_dims[n:new_n] = dims
        self._num_spiders += ndims
        self._max_dim = max(max_dim, self._max_dim)
        return range(n, new_n)

    def add_spider(self, dim: Dim) -> SpiderIdx:
        """
        Adds a spider with the given dimension to the spider graph.
        Returns the index of the new spider.
        """
        # TODO: make this more efficient
        return self.add_spiders([dim])[0]

    def add_edges(self, new_edges: Iterable[EdgeData]) -> Sequence[EdgeIdx]:
        """
        Adds edges with the given data to the spider graph.
        Returns the indices of the new edges.
        """
        new_edges = tuple(new_edges)
        if not new_edges:
            return ()
        assert self.__validate_edge_data(new_edges)
        edges, m = self._edges, self._num_edges
        nedges = len(new_edges)
        self._matrices.extend(matrix for matrix, _, _ in new_edges)
        while m + nedges > len(edges):
            edges.resize((2 * len(edges), 2), refcheck=False)
        edges[m : m + nedges, :] = [(t, h) for _, t, h in new_edges]
        self._num_edges += nedges
        return range(m, m + nedges)

    def add_edge(self, mat: Matrix, tail: SpiderIdx, head: SpiderIdx) -> EdgeIdx:
        """
        Adds an edge with the given data to the spider graph.
        Returns the index of the new edge.
        """
        # TODO: make this more efficient
        return self.add_edges([(mat, tail, head)])[0]

    def iter_edges(self) -> Iterator[EdgeData]:
        """Iterates over the edges in the spider graph."""
        for mat, (tail, head) in zip(self._matrices, self._edges):
            yield (mat, tail, head)

    def resize_capacity(
        self,
        edge_capacity: int | None = None,
        spider_capacity: int | None = None,
        max_dim: int | None = None,
    ) -> None:
        """Resizes the internal data structures of the spider graph."""
        assert self.__validate_capacities(edge_capacity, spider_capacity, max_dim)
        if edge_capacity is not None:
            self._edges.resize((edge_capacity, 2), refcheck=False)
        if spider_capacity is not None:
            self._spider_dims.resize(spider_capacity, refcheck=False)
            if (t := _get_min_dtype(spider_capacity - 1)) != self._edges.dtype:
                self._edges = self._edges.astype(t)
                self._edges_dtype_max = _UINT_DTYPE_MAX[t]
        if max_dim is not None:
            self._max_dim = max_dim
            if (t := _get_min_dtype(max_dim)) != self._spider_dims.dtype:
                self._spider_dims = self._spider_dims.astype(t)
                self._spiders_dtype_max = _UINT_DTYPE_MAX[t]

    def trim_capacity(self) -> None:
        """Trims the edge and spider capacity to the tighest possible value."""
        self._edges.resize((max(1, self._num_edges), 2))
        self._spider_dims.resize(max(1, self._num_spiders))
        # TODO: perform dtype reduction where possible

    def contraction_args(self, outputs: Iterable[int]) -> ContractionArgs:
        """
        Returns the contraction arguments for this spider graph,
        with the given selection of spiders as outputs.
        """
        outputs = tuple(outputs)
        self.__validate_spider_idxs(outputs)
        return {
            "inputs": self.edges,
            "outputs": outputs,
            "size_dict": dict(enumerate(self._spider_dims)),
        }

    def to_nx_graph(self) -> DiGraph:
        """
        Converts this spider graph to a :mod:`networkx`
        :class:`~networkx.classes.digraph.DiGraph` instance.

        Spider dimension is stored in digraph node data as the ``"dim"`` attribute.
        Edge matrix is stored in digraph edge data as the ``"mat"`` attribute.
        """
        try:
            import networkx as nx
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "SpiderGraph.to_nx_graph() requires networkx to be installed."
            ) from None
        graph = nx.DiGraph()
        # graph.add_nodes_from(self.spiders)
        for spider, dim in enumerate(self.spider_dims):
            graph.add_node(spider, dim=dim)
        # graph.add_edges_from(self.edges)
        for mat, tail, head in self.iter_edges():
            graph.add_edge(tail, head, mat=mat)
        return graph

    def __sizeof__(self) -> int:
        s = 0
        s += self._matrices.__sizeof__()
        s += sum({id(mat): mat.__sizeof__() for mat in self._matrices}.values())
        s += self._edges.__sizeof__()
        s += self._spider_dims.__sizeof__()
        s += self._num_edges.__sizeof__()
        s += self._num_spiders.__sizeof__()
        s += self._edges_dtype_max.__sizeof__()
        s += self._spiders_dtype_max.__sizeof__()
        return s

    if __debug__:

        def __validate_capacities(
            self,
            edge_capacity: int | None,
            spider_capacity: int | None,
            max_dim: int | None = None,
        ) -> Literal[True]:
            if edge_capacity is not None:
                validate(edge_capacity, int)
                if edge_capacity <= 0:
                    raise ValueError("Edge capacity must be strictly positive.")
                if edge_capacity < self._num_edges:
                    raise ValueError(
                        "Edge capacity must be >= current number of edges."
                    )
            if spider_capacity is not None:
                validate(spider_capacity, int)
                if spider_capacity <= 0:
                    raise ValueError("Spider capacity must be strictly positive.")
                if spider_capacity < self._num_spiders:
                    raise ValueError(
                        "Spider capacity must be >= current number of spiders."
                    )
            if max_dim is not None:
                validate(max_dim, int)
                if max_dim <= 0:
                    raise ValueError("Maximum dimension must be strictly positive.")
                if self._spider_dims and max_dim < np.max(self._spider_dims):
                    raise ValueError(
                        "New maximum dimension must be >= current maximum dimension."
                    )
            return True

        def __validate_spider_dims(self, dims: tuple[Dim, ...]) -> Literal[True]:
            validate(dims, tuple[Dim, ...])
            if any(dim <= 0 for dim in dims):
                raise ValueError("Dimensions must be strictly positive.")
            return True

        def __validate_spider_idxs(self, idxs: tuple[SpiderIdx, ...]) -> Literal[True]:
            validate(idxs, tuple[SpiderIdx, ...])
            n = self._num_spiders
            for idx in idxs:
                if not 0 <= idx < n:
                    raise ValueError(f"Invalid spider index {idx}.")
            return True

        def __validate_edge_data(self, edges: tuple[EdgeData, ...]) -> Literal[True]:
            validate(edges, tuple[EdgeData, ...])
            spiders, n = self._spider_dims, self._num_spiders
            for mat, tail, head in edges:
                if not 0 <= tail < n:
                    raise ValueError(f"Invalid tail spider index {tail}")
                if not 0 <= head < n:
                    raise ValueError(f"Invalid head spider index {head}")
                shape = (spiders[head], spiders[tail])
                mat_shape = autoray.shape(mat)
                if mat_shape != shape:
                    raise ValueError(
                        f"Invalid matrix shape: found {mat_shape}, expected {shape}."
                    )
            return True
