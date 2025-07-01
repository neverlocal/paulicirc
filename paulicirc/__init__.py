"""A library for quantum circuits based on Pauli gadgets."""

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

__version__ = "0.2.0"

from .utils.options import options
from .gadgets import Gadget
from .circuits import Circuit
from .builders import CircuitBuilder
from .layers import Layer

__all__ = (
    "options",
    "Gadget",
    "Layer",
    "Circuit",
    "CircuitBuilder",
    # "LayeredCircuitBuilder",
)
