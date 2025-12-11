# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Netcdf utility functions
------------------------
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset


def read_dimension(dataset: Dataset, dimension: str) -> npt.NDArray[np.float64] | None:
    """Read data axis along requested dimension"""
    try:
        return np.array(dataset.variables[dimension][:])
    except (IndexError, KeyError):
        return None


def read_group_variable(dataset: Dataset, group: str, variable: str) -> npt.NDArray[np.complex128] | None:
    """Read a variable from a group"""
    try:
        data = np.asarray(dataset.groups[group][variable][:])
        data = data["real"] + 1.0j * data["imag"]
        return data
    except (IndexError, KeyError):
        return None


def get_dataset(netcdf_file: Path, mode: str = "r") -> Dataset:
    """Get dataset from netcdf file"""
    return Dataset(netcdf_file, mode=mode)
