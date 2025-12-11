# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to handle the floating point estimation precision
-----------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class InvalidEstimationDTypePrecisionError(ValueError):
    """Handle invalid floating point precision config."""


class EstimationDTypePrecision(Enum):
    """The 32-bit or 64-bit precision."""

    BIT_32 = 1
    """Use np.float32 and np.complex64"""

    BIT_64 = 0
    """Use np.float64 and np.complex128"""


@dataclass
class EstimationDType:
    """The complex/float types used for STA_P calibration estimation."""

    complex_dtype: np.dtype
    """The type used for operating on complex numbers."""

    float_dtype: np.dtype
    """The type used for operating on real numbers."""

    def __init__(self, dtype_precision: EstimationDTypePrecision):
        """Initialize the structure."""
        if dtype_precision is EstimationDTypePrecision.BIT_32:
            self.complex_dtype = np.complex64
        elif dtype_precision is EstimationDTypePrecision.BIT_64:
            self.complex_dtype = np.complex128
        else:
            raise InvalidEstimationDTypePrecisionError(f"Invalid floating point precision '{dtype_precision}'")
        self.float_dtype = np.finfo(self.complex_dtype).dtype

    @classmethod
    def from_32bit_flag(cls, *, use_32bit_flag: bool) -> EstimationDType:
        """Return a 32-bit or 64-bit estimation types."""
        if use_32bit_flag:
            return cls(EstimationDTypePrecision.BIT_32)
        return cls(EstimationDTypePrecision.BIT_64)

    def __str__(self):
        """Format the estimation dtype."""
        nbits = int(str(np.finfo(self.float_dtype).dtype)[-2:])
        return f"float_type=float{nbits}|complex_type=complex{2 * nbits}"


def assert_numeric_types_equal(
    variable: npt.NDArray | npt.NBitBase,
    *,
    expected_dtype: np.dtype,
):
    """
    Check that the variable has the expected type.

    Parameters
    ----------
    variable: npt.NDArray | npt.NBitBase
        Variable that needs to be checked.

    expected_dtype: np.dtype
        The expected dtype.

    Raises
    ------
    AssertionError

    """
    assert variable.dtype == expected_dtype, "unexpected dtype (expected={}, got={})".format(
        np.dtype(expected_dtype), variable.dtype
    )


def assert_list_numeric_types_equal(
    variables: list[npt.NDArray | npt.NBitBase],
    *,
    expected_dtype: np.dtype,
):
    """
    Check that all objects in the iterable has expected type.

    Parameters
    ----------
    variables: list[npt.NDArray | npt.NBitBase]
        Variables that needs to be checked.

    expected_dtype: np.dtype
        The expected dtype.

    Raises
    ------
    AssertionError

    """
    if len(variables) > 0:
        assert all(v.dtype == expected_dtype for v in variables), "unexpected dtype (expected={}, got={})".format(
            np.dtype(expected_dtype), variables[0].dtype
        )
