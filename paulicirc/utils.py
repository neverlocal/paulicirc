from __future__ import annotations
from typing import Final

from optmanage import OptionManager, Option


def _validate_decimal_prec(prec: int) -> None:
    if prec < 0:
        raise ValueError("Display precisions must be >= 0.")

def _validate_atol(atol: float) -> None:
    if atol < 0.0:
        raise ValueError("Absolute tolerance must be >= 0.")

def _validate_rtol(rtol: float) -> None:
    if rtol < 0.0:
        raise ValueError("Relative tolerance must be >= 0.")

class PauliCircOptions(OptionManager):
    """
    Global options class for the :mod:`paulicirc` library.
    """

    display_prec: Option[int] = Option(int, 8, _validate_decimal_prec)
    """Number of bits of precision used when displaying phases."""

    atol: Option[float] = Option(float, 1e-5, _validate_atol)
    """Absolute tolerance used in phase equality comparison."""

    rtol: Option[float] = Option(float, 1e-8, _validate_atol)
    """Relative tolerance used in phase equality comparison."""

options: Final = PauliCircOptions()
"""
Global options for the :mod:`paulicirc` library.
"""
